module Ax2

export app

module Model

using WAV, DSP, Memoize, LRUCache, ProgressMeter, Colors, Statistics, ImageCore, ImageMorphology, DelimitedFiles, HDF5

fs_play = 48_000

load_recording(wavfile) = wavread(joinpath(datapath, wavfile))

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

function precompute_configs(nffts, nw, k, fs)
    configs = Dict{eltype(nffts),Channel{MTConfig{Float64}}}()
    for nfft in nffts
        configs[nfft] = Channel{MTConfig{Float64}}(Threads.nthreads())
        foreach(1:Threads.nthreads()) do _
            put!(configs[nfft], MTConfig{Float64}(nfft; ftest=true, nw=nw, ntapers=k, fs=fs))
        end
    end
    return configs
end

function calculate_multitaper_spectrograms(y, nffts, configs, iclip)
    @memoize LRU(maxsize=1_000_000) function _mt_pgram(idx, nfft, config)
        _y = y[idx:idx+nfft-1]
        _y .-= mean(_y)
        mt_pgram(_y, config)
    end

    mtspectrums = []
    for nfft in nffts
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
    Fs = []
    for mt in mts
        push!(Fs, hcat((Fpval(x) for x in mt)...))
    end
    return Fs
end

make_strel(t) = strel_box(t)

function refine_ftest(Fs, thresh, sigonly, morphclose, strelclose, morphopen, strelopen, minpix)
    Y_overlay = overlay(Fs)
    for j in axes(Y_overlay,2), k in axes(Y_overlay,3)
        if all(Y_overlay[1:3,j,k] .< thresh)
            Y_overlay[:,j,k] .= 1
            Y_overlay[2,j,k] = 0
        elseif sigonly
            Y_overlay[:,j,k] .= 0
            Y_overlay[4,j,k] = 1
        end
    end
    if sigonly
        if morphclose
            Y_closed = closing(Y_overlay[1,:,:], strelclose)
            Y_overlay .= reshape(Y_closed, 1, size(Y_closed)...)
            Y_overlay[2,:,:] .= 0
            Y_overlay[4,:,:] .= 1
        end
        if morphopen
            Y_opened = opening(Y_overlay[1,:,:], strelopen)
            Y_overlay .= reshape(Y_opened, 1, size(Y_opened)...)
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
    labels = label_components(F, trues(3,3))
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

function init(_datapath)
    global datapath, wavfiles
    datapath = _datapath
    wavfiles = filter(endswith(".wav"), readdir(_datapath))
end

end # module Model


module View

using ..Model, GLMakie, Preferences, Colors, FixedPointNumbers
import ..Model as M

@kwdef struct Widgets
    fig; m;
    cb_power; to; tb_nfft; cb_pval; tb_nwk; tb_thresh; cb_sigonly;
    cb_morphclose; tb_strelclose; cb_morphopen; tb_strelopen; tb_minpix;
    isl_freq;
    bt_left_big_center; bt_left_small_center; bt_right_small_center; bt_right_big_center;
    bt_left_big_width; bt_left_small_width; bt_right_small_width; bt_right_big_width;
    sl_time_center; sl_time_width;
    lb_status; bt_play; bt_csv; bt_hdf;
    ax; hm; hm_pvals; ax1; li1; ax2; li2; ax3; li3;
end

@kwdef struct Observables
    y; fs; nffts; noverlaps; nw; k; thresh; strelclose; strelopen; minpix;
    Ys; Y; Y_freq; Y_time; configs; ifreq; itime; iclip; mtspectrums; Y_MTs; Y_MT; Fs; F;
    alpha_power; alpha_pval; powers; freqs; times; freqs_mt; times_mt; pvals; cr;
    iclip_subsampled; y_clip; times_yclip;
    cumpowers1; cumpowers1_freqs; cumpowers2; times_cumpowers2;
end

hz2khz = 1000
display_size = (3000,2000)
max_width_sec = 10

pref_defaults = (;
    figsize = (640,450),
    isl_freq = missing,
    sl_time_center = 0,
    sl_time_width = missing,
    m = missing,
    cb_tooltips = false,
    cb_power = true,
    to = true,
    tb_nfft = "512",
    cb_pval = false,
    tb_nwk = "4,6",
    tb_thresh = "0.01",
    cb_sigonly = false,
    cb_morphclose = false,
    cb_morphopen = false,
    tb_strelclose = "3x9",
    tb_strelopen = "3x9",
    tb_minpix = "0",
    )

function init()
    _figsize_pref = @load_preference("figsize", pref_defaults.figsize)
    figsize_pref = typeof(_figsize_pref)<:Tuple ? _figsize_pref : eval(Meta.parse(_figsize_pref))
    _isl_freq_pref = @load_preference("isl_freq", pref_defaults.isl_freq)
    isl_freq_pref = ismissing(_isl_freq_pref) ? _isl_freq_pref : eval(Meta.parse(_isl_freq_pref))
    sl_time_center_pref = @load_preference("sl_time_center", pref_defaults.sl_time_center)
    sl_time_width_pref = @load_preference("sl_time_width", pref_defaults.sl_time_width)
    m_pref = @load_preference("m", pref_defaults.m)
    cb_tooltips_pref = @load_preference("cb_tooltips", pref_defaults.cb_tooltips)
    cb_power_pref = @load_preference("cb_power", pref_defaults.cb_power)
    to_pref = @load_preference("to", pref_defaults.to)
    tb_nfft_pref = @load_preference("tb_nfft", pref_defaults.tb_nfft)
    cb_pval_pref = @load_preference("cb_pval", pref_defaults.cb_pval)
    tb_nwk_pref = @load_preference("tb_nwk", pref_defaults.tb_nwk)
    tb_thresh_pref = @load_preference("tb_thresh", pref_defaults.tb_thresh)
    cb_sigonly_pref = @load_preference("cb_sigonly", pref_defaults.cb_sigonly)
    cb_morphclose_pref = @load_preference("cb_morphclose", pref_defaults.cb_morphclose)
    tb_strelclose_pref = @load_preference("tb_strelclose", pref_defaults.tb_strelclose)
    cb_morphopen_pref = @load_preference("cb_morphopen", pref_defaults.cb_morphopen)
    tb_strelopen_pref = @load_preference("tb_strelopen", pref_defaults.tb_strelopen)
    tb_minpix_pref = @load_preference("tb_minpix", pref_defaults.tb_minpix)

    fig = Figure(size=figsize_pref)

    m = Menu(fig[1,1:3], options=M.wavfiles, default=coalesce(m_pref, M.wavfiles[1]))

    y_fs_ = @lift M.load_recording($(m.selection))
    y = @lift $(y_fs_)[1]
    fs = @lift $(y_fs_)[2]

    gl_tt = GridLayout(fig[2,3])
    Label(gl_tt[1,1, Bottom()], "tooltips", tellheight=false, tellwidth=false)
    cb_tooltips = Checkbox(gl_tt[2,1, Top()], checked = cb_tooltips_pref,
                           tellheight=false, tellwidth=false)

    gl_fft = GridLayout(fig[1:3,4])
    Label(gl_fft[1,1, Top()], "power")
    cb_power = Checkbox(gl_fft[1,1], checked = cb_power_pref)
    Label(gl_fft[2,1, Top()], "Hanning")
    Label(gl_fft[2,1, Bottom()], "Slepian")
    to = Toggle(gl_fft[2,1], active=to_pref, orientation=:vertical)
    Label(gl_fft[3,1,Top()], "nfft")
    tb_nfft = Textbox(gl_fft[3,1], stored_string=tb_nfft_pref,
                      validator = s -> all(isdigit(c) || c==',' for c in s))

    Label(gl_fft[4,1, Top()], "p-val")
    cb_pval = Checkbox(gl_fft[4,1], checked = cb_pval_pref)
    Label(gl_fft[5,1,Top()], "NW,K")
    tb_nwk = Textbox(gl_fft[5,1], stored_string=tb_nwk_pref,
                     validator = s -> all(isdigit(c) || in(c,".,") for c in s))
    Label(gl_fft[6,1,Top()], "threshold")
    tb_thresh = Textbox(gl_fft[6,1], stored_string=tb_thresh_pref, validator=Float64)
    Label(gl_fft[7,1, Top()], "sig. only")
    cb_sigonly = Checkbox(gl_fft[7,1], checked = cb_sigonly_pref)

    gl_morph = GridLayout(fig[1:3,5])
    Label(gl_morph[1,1, Top()], "morph.\nclose")
    cb_morphclose = Checkbox(gl_morph[1,1], checked = cb_morphclose_pref)
    Label(gl_morph[2,1, Top()], "str. el.")
    tb_strelclose = Textbox(gl_morph[2,1], stored_string=tb_strelclose_pref,
                            validator = s -> all(isdigit(c) || c=='x' for c in s))
    Label(gl_morph[3,1, Top()], "morph.\nopen")
    cb_morphopen = Checkbox(gl_morph[3,1], checked = cb_morphopen_pref)
    Label(gl_morph[4,1, Top()], "str. el.")
    tb_strelopen = Textbox(gl_morph[4,1], stored_string=tb_strelopen_pref,
                           validator = s -> all(isdigit(c) || c=='x' for c in s))

    Label(gl_morph[5,1, Top()], "min. pix")
    tb_minpix = Textbox(gl_morph[5,1], stored_string=tb_minpix_pref,
                        validator = s -> all(isdigit(c) for c in s))

    nffts = @lift parse.(Int, split($(tb_nfft.stored_string), ','))
    noverlaps = @lift div.($nffts, 2)
    nw = @lift parse(Float64, split($(tb_nwk.stored_string), ',')[1])
    k = @lift parse(Int, split($(tb_nwk.stored_string), ',')[2])
    thresh = @lift parse(Float64, $(tb_thresh.stored_string))
    strelclose = @lift M.make_strel(tuple(parse.(Int, split($(tb_strelclose.stored_string), 'x'))...))
    strelopen = @lift M.make_strel(tuple(parse.(Int, split($(tb_strelopen.stored_string), 'x'))...))
    minpix = @lift parse(Int, $(tb_minpix.stored_string))

    Ys = @lift M.calculate_hanning_spectrograms($y, $nffts, $fs)
    Y = @lift M.overlay($Ys, M.dB)

    Y_freq = @lift M.freq($Ys[argmax($nffts)])
    Y_time = @lift M.time($Ys[argmin($nffts)])

    configs = @lift M.precompute_configs($nffts, $nw, $k, $fs)

    isl_freq = IntervalSlider(fig[3,1], range=0:0.01:1, horizontal=false,
                              startvalues = coalesce(isl_freq_pref, tuple(0, 1)))
    gl_pan = GridLayout(fig[4,3:4], halign=:left)
    bt_left_big_center = Button(gl_pan[1,1], label="<<")
    bt_left_small_center = Button(gl_pan[1,2], label="<")
    bt_right_small_center = Button(gl_pan[1,3], label=">")
    bt_right_big_center = Button(gl_pan[1,4], label=">>")
    Label(fig[4,2, Left()], "center")
    step = (nffts[][1]-noverlaps[][1]) / length(y[])
    sl_time_center = Slider(fig[4,2], range=0:step:1, startvalue=sl_time_center_pref)

    gl_zoom = GridLayout(fig[5,3:4], halign=:left)
    bt_left_big_width = Button(gl_zoom[1,1], label="<<")
    bt_left_small_width = Button(gl_zoom[1,2], label="<")
    bt_right_small_width = Button(gl_zoom[1,3], label=">")
    bt_right_big_width = Button(gl_zoom[1,4], label=">>")
    Label(fig[5,2, Left()], "width")
    maxvalue = max_width_sec * fs[] / length(y[])
    sl_time_width = Slider(fig[5,2], range=0:step:maxvalue,
                           startvalue = coalesce(sl_time_width_pref, maxvalue))

    on(_->set_close_to!(sl_time_center, sl_time_center.value[] - sl_time_width.value[] / 2),
       bt_left_big_center.clicks)
    on(_->set_close_to!(sl_time_center, sl_time_center.value[] - sl_time_width.value[] / 10),
       bt_left_small_center.clicks)
    on(_->set_close_to!(sl_time_center, sl_time_center.value[] + sl_time_width.value[] / 10),
       bt_right_small_center.clicks)
    on(_->set_close_to!(sl_time_center, sl_time_center.value[] + sl_time_width.value[] / 2),
       bt_right_big_center.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*0.5),
       bt_left_big_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*0.9),
       bt_left_small_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*1.1),
       bt_right_small_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*1.5),
       bt_right_big_width.clicks)

    lb_status = Label(fig[7,1:4], " ")

    gl_out = GridLayout(fig[6,3:4], tellheight=false, tellwidth=false)

    bt_play = Button(gl_out[1,1], label="play")
    on(_->M.play(y[], iclip[], fs[]), bt_play.clicks)

    bt_csv = Button(gl_out[1,2], label="CSV")
    on(bt_csv.clicks) do _
        if !cb_pval.checked[] || !cb_sigonly.checked[]
            lb_status.text[] = "p-val and sig. only must both be checked to output CSV"
            return
        end
        filename = joinpath(M.datapath, string(m.selection[], '-', iclip[][1], '-', iclip[][2], ".csv"))
        M.save_csv(filename, F[], Y_freq[], ifreq[], Y_time[], itime[])
        lb_status.text[] = "CSV saved to $filename"
    end

    bt_hdf = Button(gl_out[1,3], label="HDF")
    on(bt_hdf.clicks) do _
        if !cb_pval.checked[] || !cb_sigonly.checked[]
            lb_status.text[] = "p-val and sig. only must both be checked to output HDF"
            return
        end
        filename = joinpath(M.datapath, string(m.selection[], '-', iclip[][1], '-', iclip[][2], ".hdf"))
        M.save_hdf(filename, F[], Y_freq[], ifreq[], Y_time[], itime[])
        lb_status.text[] = "HDF saved to $filename"
    end

    # indices into Y
    ifreq = lift(isl_freq.interval, Y_freq) do x, Y_freq
        start = 1 + round(Int, (length(Y_freq)-1) * x[1])
        stop = 1 + round(Int, (length(Y_freq)-1) * x[2])
        step = max(1, fld(stop-start+1, display_size[2]))
        start:step:stop
    end
    itime = lift(sl_time_center.value, sl_time_width.value, Y_time) do c, w, Y_time
        frac2fft(x) = round(Int, x*length(Y_time))
        start = max(1, frac2fft(c-w))
        step = max(1, Int(fld(frac2fft(w), display_size[1])))
        stop = min(length(Y_time), frac2fft(c+w))
        start:step:stop
    end

    # indices into y
    iclip = lift(sl_time_center.value, sl_time_width.value, y) do c, w, y
        frac2tic(x) = round(Int, x*length(y))
        (max(1, frac2tic(c-w)), min(length(y), frac2tic(c+w)))
    end

    mtspectrums = @lift begin
        if !$(to.active) || $(cb_pval.checked)
            M.calculate_multitaper_spectrograms($y, $nffts, $configs, $iclip)
        else
            fill(Vector{M.Periodograms.PeriodogramF}(undef, 0), 0)
        end
    end

    Y_MTs = @lift begin
        if !$(to.active)
            M.coalesce_multitaper_power($mtspectrums)
        else
            fill(M.DSP.Periodograms.Spectrogram(Matrix{Float64}(undef, 0, 0), 0:0., 0:0.), 0)
        end
    end
    Y_MT = @lift $(to.active) ? Array{Float32}(undef, 0, 0, 0) : M.overlay($Y_MTs, M.dB)

    Fs = @lift begin
        if $(cb_pval.checked)
            M.coalesce_multitaper_ftest($mtspectrums)
        else
            fill(Matrix{Float64}(undef, 0, 0), 0)
        end
    end
    F = @lift begin
        if $(cb_pval.checked)
            M.refine_ftest($Fs, $thresh, $(cb_sigonly.checked),
                           $(cb_morphclose.checked), $strelclose,
                           $(cb_morphopen.checked), $strelopen,
                           $minpix)
        else
            Matrix{RGBA{N0f8}}(undef, 0, 0)
        end
    end

    alpha_power = @lift $(cb_power.checked) * 0.5 + !$(cb_pval.checked) * 0.5
    alpha_pval = @lift $(cb_pval.checked) * 0.5 + !$(cb_power.checked) * 0.5

    powers = @lift begin
        if $(to.active)
            all(in.(extrema($itime), Ref(axes($Y,3)))) || return RGBA{N0f8}[1 0; 0 0]
            all(in.(extrema($ifreq), Ref(axes($Y,2)))) || return RGBA{N0f8}[1 0; 0 0]
            Y_scratch = $Y[:,$ifreq,$itime]
        else
            all(in.(extrema($ifreq), Ref(axes($Y_MT,2)))) || return RGBA{N0f8}[1 0; 0 0]
            Y_scratch = $Y_MT[:, $ifreq, 1:$itime.step:end]
        end
        M.scale_and_color(Y_scratch)
    end

    freqs = @lift tuple($Y_freq[$ifreq][[1,end]] ./ hz2khz...)
    times = @lift tuple($Y_time[$itime][[1,end]]...)
    ax,hm = image(fig[3,2], times, freqs, powers;
                  interpolate=false, alpha=alpha_power, visible=cb_power.checked,
                  inspector_label = (pl,i,pos)->string(
                          "time = ", pos[1], " sec\n",
                          "freq = ", pos[2], " Hz\n",
                          "power = ", red(pos[3]).i+0, ',', green(pos[3]).i+0, ',', blue(pos[3]).i+0))

    ax.xlabel[] = "time (s)"
    ax.ylabel[] = "frequency (kHz)"
    onany(freqs, times) do f,t
        limits!(ax, t..., f...)
    end

    freqs_mt = @lift $(cb_pval.checked) ? $freqs : tuple(0.,0.)
    times_mt = @lift $(cb_pval.checked) ? $times : tuple(0.,0.)
    pvals = @lift begin
        if $(cb_pval.checked)
            all(in.(extrema($ifreq), Ref(axes($F,1)))) || return RGBA{N0f8}[1 0; 0 0]
            $F'[1:$itime.step:end, $ifreq]
        else
            Matrix{RGBA{N0f8}}(undef, 1, 1)
        end
    end
    cr = @lift ($thresh,1)
    hm_pvals = image!(times_mt, freqs_mt, pvals;
                      interpolate=false,
                      colormap=:grays, colorrange=cr, lowclip=(:fuchsia, 1),
                      alpha=alpha_pval,
                      visible=cb_pval.checked,
                      inspector_label = (pl,i,pos)->string(
                              "time = ", pos[1], " sec\n",
                              "freq = ", pos[2], " Hz\n",
                              "power = ", red(pos[3]).i+0, ',', green(pos[3]).i+0, ',', blue(pos[3]).i+0))

    iclip_subsampled = @lift $iclip[1] : max(1, fld($iclip[2]-$iclip[1], display_size[2])) : $iclip[2]
    y_clip = @lift view(y[], $iclip_subsampled)
    times_yclip = @lift Point2f.(zip($iclip_subsampled ./ $fs, $y_clip))

    ax1, li1 = lines(fig[6,2], times_yclip,
                     inspector_label = (pl,i,pos)->string("time = ", pos[1], " sec\n",
                                                          "amplitude = ", pos[2], " V"))
    ax1.xticklabelsvisible[] = ax1.yticklabelsvisible[] = false
    ax1.ylabel[] = "amplitude"
    onany((yc,ics,fs)->limits!(ax1, ics[1]/fs, ics[end]/fs, extrema(yc)...),
          y_clip, iclip_subsampled, fs)

    cumpowers1 = @lift M.cumpower($powers, 1)
    cumpowers1_freqs = @lift Point2f.(zip($cumpowers1, $Y_freq[$ifreq]))

    ax2, li2 = lines(fig[3,3], cumpowers1_freqs,
                     inspector_label = (pl,i,pos)->string("freq = ", pos[2], " Hz\n",
                                                          "power = ", pos[1], " dB"))
    ax2.xticklabelsvisible[] = ax2.yticklabelsvisible[] = false
    ax2.xlabel[] = "power"
    onany((cp,Yf,i)->limits!(ax2, extrema(cp)..., Yf[i[1]], Yf[i[end]]),
          cumpowers1, Y_freq, ifreq)

    cumpowers2 = @lift M.cumpower($powers, 2)
    times_cumpowers2 = @lift Point2f.(zip($Y_time[$itime], $cumpowers2))

    ax3, li3 = lines(fig[2,2], times_cumpowers2,
                     inspector_label = (pl,i,pos)->string("time = ", pos[1], " sec\n",
                                                          "power = ", pos[2], " dB"))
    ax3.xticklabelsvisible[] = ax3.yticklabelsvisible[] = false
    ax3.ylabel[] = "power"
    onany((cp,Yt,i)->limits!(ax3, Yt[i[1]], Yt[i[end]], extrema(cp)...),
          cumpowers2, Y_time, itime)

    colsize!(fig.layout, 2, Auto(8))
    colsize!(fig.layout, 3, Auto(1))
    rowsize!(fig.layout, 2, Auto(1))
    rowsize!(fig.layout, 3, Auto(4))
    rowsize!(fig.layout, 6, Auto(1))

    tooltip!(tb_nwk,
             """
             K is # of tapers
             NW is the spectral bandwidth
             K should be less than 2NW-1
             """,
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_nfft,
             """
             short time windows can resolve quickly varying signals better
             """,
             placement = :left, enabled = cb_tooltips.checked)

    for cb in (cb_tooltips, cb_power, cb_pval, cb_sigonly, cb_morphclose, cb_morphopen)
        foreach(x->x.inspectable[]=false, cb.blockscene.plots)
    end
    DataInspector(enabled=cb_tooltips.checked)

    on(x->@set_preferences!("figsize"=>string(tuple(x.widths...))), fig.scene.viewport)
    on(x->@set_preferences!("isl_freq"=>string(x)), isl_freq.interval)
    on(x->@set_preferences!("sl_time_center"=>x), sl_time_center.value)
    on(x->@set_preferences!("sl_time_width"=>x), sl_time_width.value)
    on(x->@set_preferences!("m"=>x), m.selection)
    on(x->@set_preferences!("cb_tooltips"=>x), cb_tooltips.checked)
    on(x->@set_preferences!("cb_power"=>x), cb_power.checked)
    on(x->@set_preferences!("to"=>x), to.active)
    on(x->@set_preferences!("tb_nfft"=>x), tb_nfft.stored_string)
    on(x->@set_preferences!("cb_pval"=>x), cb_pval.checked)
    on(x->@set_preferences!("tb_nwk"=>x), tb_nwk.stored_string)
    on(x->@set_preferences!("tb_thresh"=>x), tb_thresh.stored_string)
    on(x->@set_preferences!("cb_sigonly"=>x), cb_sigonly.checked)
    on(x->@set_preferences!("cb_morphclose"=>x), cb_morphclose.checked)
    on(x->@set_preferences!("tb_strelclose"=>x), tb_strelclose.stored_string)
    on(x->@set_preferences!("cb_morphopen"=>x), cb_morphopen.checked)
    on(x->@set_preferences!("tb_strelopen"=>x), tb_strelopen.stored_string)
    on(x->@set_preferences!("tb_minpix"=>x), tb_minpix.stored_string)

    widgets = Widgets(
        fig, m,
        cb_power, to, tb_nfft, cb_pval, tb_nwk, tb_thresh, cb_sigonly,
        cb_morphclose, tb_strelclose, cb_morphopen, tb_strelopen, tb_minpix,
        isl_freq,
        bt_left_big_center, bt_left_small_center, bt_right_small_center, bt_right_big_center,
        bt_left_big_width, bt_left_small_width, bt_right_small_width, bt_right_big_width,
        sl_time_center, sl_time_width,
        lb_status, bt_play, bt_csv, bt_hdf,
        ax, hm, hm_pvals, ax1, li1, ax2, li2, ax3, li3
        )

    observables = Observables(
        y, fs, nffts, noverlaps, nw, k, thresh, strelclose, strelopen, minpix,
        Ys, Y, Y_freq, Y_time, configs, ifreq, itime, iclip, mtspectrums, Y_MTs, Y_MT, Fs, F,
        alpha_power, alpha_pval, powers, freqs, times, freqs_mt, times_mt, pvals, cr,
        iclip_subsampled, y_clip, times_yclip,
        cumpowers1, cumpowers1_freqs, cumpowers2, times_cumpowers2,
        )

    return widgets, observables
end

end # module View


using .Model, .View

function app(datapath)
    Model.init(datapath)
    widgets, observables = View.init()
    notify(observables.freqs)
    notify(observables.cumpowers1)
    notify(observables.cumpowers2)
    notify(observables.y_clip)
    notify(observables.nffts)
    display(widgets.fig)
    return widgets, observables
end

end # module Ax2
