module Ax2

using WAV, DSP, LRUCache, ProgressMeter, Colors, Statistics, ImageCore, ImageMorphology, ImageFiltering, DelimitedFiles, HDF5

export load_recording, calculate_hanning_spectrograms, dB, overlay, precompute_configs, calculate_multitaper_spectrograms, coalesce_multitaper_power, coalesce_multitaper_ftest, make_strel, refine_ftest, scale_and_color, cumpower, play, get_components, save_csv, save_hdf
export freq, time, Periodograms
export app

fs_play = 48_000

load_recording(wavfile) = wavread(wavfile)

Ys_cache = LRU{Int,DSP.Periodograms.Spectrogram}(maxsize=10)

function calculate_hanning_spectrograms(y, nffts, fs)
    Ys = Vector{DSP.Periodograms.Spectrogram}(undef, length(nffts))
    Threads.@threads :greedy for (i,nfft) in enumerate(nffts)
        Ys[i] = get!(()->spectrogram.(Ref(y[:,1]), nfft; fs=fs, window=hanning), Ys_cache, nfft)
    end
    return Ys
end

dB = x->20*log10.(power(x))
overlay(Ys, f) = overlay(f.(Ys))
function overlay(Ys)
    ntime, nfreq = size.(Ys,2), size.(Ys,1)
    mintime = minimum(round.(Int, maximum(ntime) ./ ntime) .* ntime)
    minfreq = minimum(round.(Int, maximum(nfreq) ./ nfreq) .* nfreq)
    Y_overlay = zeros(Float32, 4, minfreq, mintime)
    for (icolor, Yi) in enumerate(Ys)
        sz = size(Yi)
        scale = round.(Int, (minfreq, mintime) ./ sz)
        for f0 in 1:scale[1], t0 in 1:scale[2]
            fdelta = sz[1] - length(f0:scale[1]:minfreq)
            tdelta = sz[2] - length(t0:scale[2]:mintime)
            Y_overlay[icolor,
              f0 : scale[1] : end,
              t0 : scale[2] : end] .= Yi[1:end-fdelta, 1:end-tdelta]
        end
    end
    Y_overlay[4,:,:] .= 1
    if length(Ys) == 2
        Y_overlay[3,:,:] .= Y_overlay[2,:,:]
        Y_overlay[2,:,:] .= Y_overlay[1,:,:]
    elseif length(Ys) == 1
        Y_overlay[2,:,:] .= Y_overlay[3,:,:] .= Y_overlay[1,:,:]
    end
    return Y_overlay
end

configs = Dict{Int64,Channel{MTConfig{Float64}}}()

function precompute_config(nfft, nw, k, fs)
    configs[nfft] = Channel{MTConfig{Float64}}(Threads.nthreads())
    foreach(1:Threads.nthreads()) do _
        put!(configs[nfft], MTConfig{Float64}(nfft; ftest=true, nw=nw, ntapers=k, fs=fs))
    end
end

function calculate_multitaper_spectrograms(y, nffts, nw, k, fs, iclip)
    function _mt_pgram(idx, nfft, config)
        _y = y[idx:idx+nfft-1]
        _y .-= mean(_y)
        mt_pgram(_y, config)
    end

    mtspectrums = []
    for nfft in nffts
        haskey(configs, nfft) || precompute_config(nfft, nw, k, fs)
        noverlap = div(nfft, 2)
        idxs = iclip[1] : nfft-noverlap : iclip[2]+1-nfft+1
        mtspectrum = Vector{Periodograms.PeriodogramF}(undef, length(idxs))
        i_idxs = Channel() do chnl
            foreach(i_idx->put!(chnl,i_idx), enumerate(idxs))
        end
        p = Progress(length(idxs))
        @sync for _ in 1:Threads.nthreads()
            Threads.@spawn begin
                config = take!(configs[nfft])
                for (i,idx) in i_idxs
                    mtspectrum[i] = _mt_pgram(idx, nfft, config)
                    next!(p)
                end
                put!(configs[nfft], config)
            end
        end
        finish!(p)
        push!(mtspectrums, mtspectrum)
    end
    return mtspectrums
end

function coalesce_multitaper_power(mts)
    Y_MTs = []
    for mt in mts
        p = hcat((power(x) for x in mt)...)
        push!(Y_MTs, DSP.Periodograms.Spectrogram(p, 0:0., 0:0.))
    end
    return Y_MTs
end

function coalesce_multitaper_ftest(mts)
    Fs = Matrix{Float64}[]
    for mt in mts
        push!(Fs, hcat((Fpval(x) for x in mt)...))
    end
    return Fs
end

make_strel(t) = strel_box(t)

function get_pad(pad, strel, Y, val)
    if pad
        padlo = tuple(-[first.(axes(strel))...]...)
        padhi = last.(axes(strel))
        Y_padded = padarray(view(Y,1,:,:), Fill(val, padlo, padhi))
    else
        padlo = padhi = (0,0)
        Y_padded = Y[1,:,:]
    end
    return Y_padded, [1:1, 1+padlo[1]:size(Y,2)+padlo[1], 1+padlo[2]:size(Y,3)+padlo[2]]
end

function refine_ftest(Fs, pval, sigonly, morphclose, strelclose, morphopen, strelopen,
                      minpix, pad=true)
    Y_overlay = overlay(Fs)
    for j in axes(Y_overlay,2)
        Threads.@threads for k in axes(Y_overlay,3)
            if all(x->x<pval, view(Y_overlay,1:3,j,k))
                Y_overlay[:,j,k] .= 1
                Y_overlay[2,j,k] = 0
            elseif sigonly
                Y_overlay[:,j,k] .= 0
                Y_overlay[4,j,k] = 1
            end
        end
    end
    if sigonly
        if morphclose
            Y_padded, ipad = get_pad(pad, strelclose, Y_overlay, 0)
            Y_closed = closing(Y_padded, strelclose)
            Y_overlay .= reshape(Y_closed, 1, size(Y_closed)...)[ipad...]
            Y_overlay[2,:,:] .= 0
            Y_overlay[4,:,:] .= 1
        end
        if morphopen
            Y_padded, ipad = get_pad(pad, strelopen, Y_overlay, 1)
            Y_opened = opening(Y_padded, strelopen)
            Y_overlay .= reshape(Y_opened, 1, size(Y_opened)...)[ipad...]
            Y_overlay[2,:,:] .= 0
            Y_overlay[4,:,:] .= 1
        end
        if minpix>0
            labels = label_components(Y_overlay[1,:,:], trues(3,3))
            indices = component_indices(labels)
            for idx in Iterators.filter(x->length(x) < minpix, Iterators.drop(indices, 1))
                view(Y_overlay, 1,:,:)[idx] .= 0
                view(Y_overlay, 2,:,:)[idx] .= 0
                view(Y_overlay, 3,:,:)[idx] .= 0
            end
        end
    end
    Y_converted = N0f8.(Y_overlay)
    dropdims(collect(reinterpret(RGBA{N0f8}, Y_converted)), dims=1)
end

function scale_and_color(Y_overlay)
    q = quantile(Y_overlay[1:3,:,:], [0.01,0.99])
    f = scaleminmax(N0f8, q...)
    Y_scaled = f.(Y_overlay)
    Y_scaled[4,:,:] .= 1
    collect(dropdims(reinterpret(RGBA{N0f8}, Y_scaled), dims=1)')
end

function cumpower(p,d)
    x = dropdims(sum(p, dims=d), dims=d)
    @. red(x) + green(x) + blue(x)
end

function play(y, iclip, fs)
    yfilt = filtfilt(digitalfilter(Lowpass(fs_play/2/fs[]), Butterworth(4)),
                     y[iclip[1]:iclip[2], 1])
    ydown = resample(yfilt, fs_play/fs)
    wavplay(ydown, fs_play)
end

function get_components(F, Y_freq, ifreq, Y_time, itime)
    labels = label_components(view(F,ifreq,:), trues(3,3))
    indices = component_indices(CartesianIndex, labels)
    data = Matrix{Float64}(undef, length(indices)-2, 4)
    for (i,idx) in Iterators.drop(enumerate(indices), 2)
        bb = extrema(idx)
        lo, hi = Y_freq[ifreq][[bb[1].I[1], bb[2].I[1]]]
        start, stop = Y_time[itime][[bb[1].I[2], bb[2].I[2]]]
        data[i-2,:] .= (start, stop, lo, hi)
    end
    return data
end

function save_csv(filename, F, Y_freq, ifreq, Y_time, itime)
    data = get_components(F, Y_freq, ifreq, Y_time, itime)
    writedlm(filename, ["start (sec)" "stop (sec)" "low (Hz)" "high (Hz)"; data], ',')
end

function save_hdf(filename, F, Y_freq, ifreq, Y_time, itime)
    data = get_components(F, Y_freq, ifreq, Y_time, itime)
    fid = h5open(filename, "w")
    @write fid data
    HDF5.attributes(fid["data"])["header"] = "start (sec), stop (sec), low (Hz), high (Hz)"
    close(fid)
end

app() = nothing

end # module Ax2
