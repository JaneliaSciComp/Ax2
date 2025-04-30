module SentimentAnalysis

using WAV, DSP, GLMakie, Multitaper, Memoize, LRUCache, Preferences, ProgressMeter, Colors, Statistics, ImageCore

export gui

function gui(datapath)
    pref_defaults = (;
        figsize = (640,450),
        isl_freq = missing,
        sl_time_center = 0,
        sl_time_width = missing,
        m = missing,
        cb_pval = false,
        cb_sigonly = false,
        tb_thresh = "0.01",
        cb_power = true,
        to = true,
        tb_nfft = "512",
        )

    _figsize_pref = @load_preference("figsize", pref_defaults.figsize)
    figsize_pref = typeof(_figsize_pref)<:Tuple ? _figsize_pref : eval(Meta.parse(_figsize_pref))
    _isl_freq_pref = @load_preference("isl_freq", pref_defaults.isl_freq)
    isl_freq_pref = ismissing(_isl_freq_pref) ? _isl_freq_pref : eval(Meta.parse(_isl_freq_pref))
    sl_time_center_pref = @load_preference("sl_time_center", pref_defaults.sl_time_center)
    sl_time_width_pref = @load_preference("sl_time_width", pref_defaults.sl_time_width)
    m_pref = @load_preference("m", pref_defaults.m)
    cb_pval_pref = @load_preference("cb_pval", pref_defaults.cb_pval)
    cb_sigonly_pref = @load_preference("cb_sigonly", pref_defaults.cb_sigonly)
    tb_thresh_pref = @load_preference("tb_thresh", pref_defaults.tb_thresh)
    cb_power_pref = @load_preference("cb_power", pref_defaults.cb_power)
    to_pref = @load_preference("to", pref_defaults.to)
    tb_nfft_pref = @load_preference("tb_nfft", pref_defaults.tb_nfft)

    hz2khz = 1000
    fs_play = 48_000
    nw, k = 4.0, 6
    display_size = (3000,2000)
    max_width_sec = 60

    fig = Figure(size=figsize_pref)

    wavfiles = filter(endswith(".wav"), readdir(datapath))
    m = Menu(fig[1,1:3], options=wavfiles, default=coalesce(m_pref, wavfiles[1]))

    y_fs_ = @lift wavread(joinpath(datapath, $(m.selection)))
    y = @lift $(y_fs_)[1]
    fs = @lift $(y_fs_)[2]

    gl = GridLayout(fig[1:4,4])
    Label(gl[1,1, Top()], "p-val")
    cb_pval = Checkbox(gl[1,1], checked = cb_pval_pref)
    Label(gl[2,1, Top()], "sig. only")
    cb_sigonly = Checkbox(gl[2,1], checked = cb_sigonly_pref)
    Label(gl[3,1,Top()], "threshold")
    tb_thresh = Textbox(gl[3,1], stored_string=tb_thresh_pref, validator=Float64)
    Label(gl[4,1, Top()], "power")
    cb_power = Checkbox(gl[4,1], checked = cb_power_pref)
    Label(gl[5,1, Top()], "Hanning")
    Label(gl[5,1, Bottom()], "Slepian")
    to = Toggle(gl[5,1], active=to_pref, orientation=:vertical)
    Label(gl[6,1,Top()], "nfft")
    tb_nfft = Textbox(gl[6,1], stored_string=tb_nfft_pref,
                      validator = s -> all(isdigit(c) || c==',' for c in s))
    bt_play = Button(gl[7,1], label="play")

    thresh = @lift parse(Float64, $(tb_thresh.stored_string))
    nffts = @lift parse.(Int, split($(tb_nfft.stored_string), ','))
    noverlaps = @lift div.($nffts, 2)

    function overlay(Ys, f, p)
        ntime, nfreq = size.(f.(Ys),2), size.(f.(Ys),1)
        mintime = minimum(round.(Int, maximum(ntime) ./ ntime) .* ntime)
        minfreq = minimum(round.(Int, maximum(nfreq) ./ nfreq) .* nfreq)
        Y_scratch = zeros(Float32, 4, minfreq, mintime)
        for (icolor, Yi) in enumerate(Ys)
            sz = size(f(Yi))
            scale = round.(Int, (minfreq, mintime) ./ sz)
            for f0 in 1:scale[1], t0 in 1:scale[2]
                fdelta = sz[1] - length(f0:scale[1]:minfreq)
                tdelta = sz[2] - length(t0:scale[2]:mintime)
                Y_scratch[icolor,
                  f0 : scale[1] : end,
                  t0 : scale[2] : end] .= f(Yi)[1:end-fdelta, 1:end-tdelta]
            end
        end
        q = quantile(Y_scratch, p)
        f = scaleminmax(N0f8, q...)
        Y_scaled = f.(Y_scratch)
        Y_scaled[4,:,:] .= 1
        if length(Ys) == 2
            Y_scaled[3,:,:] .= Y_scaled[2,:,:]
            Y_scaled[2,:,:] .= Y_scaled[1,:,:]
        elseif length(Ys) == 1
            Y_scaled[2,:,:] .= Y_scaled[3,:,:] .= Y_scaled[1,:,:]
        end
        dropdims(collect(reinterpret(RGBA{N0f8}, Y_scaled)), dims=1)
    end

    Ys_cache = LRU{Int,DSP.Periodograms.Spectrogram}(maxsize=10)
    Ys = lift(y, nffts, fs) do y, nffts, fs
        Ys = DSP.Periodograms.Spectrogram[]
        l = ReentrantLock()
        Threads.@threads for nfft in nffts
            _Y = get!(()->spectrogram.(Ref(y[:,1]), nfft; fs=fs, window=hanning),
                      Ys_cache, nfft)
            lock(l);  push!(Ys, _Y);  unlock(l)
        end
        return Ys
    end
    Y = @lift overlay($Ys, x->20*log10.(power(x)), [0.01,0.99])

    Y_freq = @lift freq($Ys[argmax($nffts)])
    Y_time = @lift time($Ys[argmin($nffts)])

    tapers = @lift dpss_tapers.($nffts, nw, k, :tap)

    @memoize LRU(maxsize=100_000) _multispec(idx, n, NW, K, dt, dpVec, Ftest) =
            multispec((@view y[][idx:idx+n-1]), NW=NW, K=K, dt=dt, dpVec=dpVec, Ftest=Ftest)
    function multitaper_spectrogram(i1, i2, n; fs=1, nw=4.0, k=6, tapers=tapers)
        noverlap = div(n, 2)
        idxs = i1 : n-noverlap : i2+1-n+1
        mtspectrum = Vector{MTSpectrum}(undef, length(idxs))
        @showprogress dt=1 Threads.@threads :greedy for (i,idx) in enumerate(idxs)
            mtspectrum[i] = _multispec(idx, n, nw, k, 1/fs, tapers, true)
        end
        return mtspectrum
    end

    isl_freq = IntervalSlider(fig[3:4,1], range=0:0.01:1, horizontal=false,
                              startvalues = coalesce(isl_freq_pref, tuple(0, 1)))
    gl2 = GridLayout(fig[5,3:4])
    bt_left_big_center = Button(gl2[1,1], label="<<")
    bt_left_small_center = Button(gl2[1,2], label="<")
    bt_right_small_center = Button(gl2[1,3], label=">")
    bt_right_big_center = Button(gl2[1,4], label=">>")
    Label(fig[5,2, Left()], "center")
    step = (nffts[][1]-noverlaps[][1]) / length(y[])
    sl_time_center = Slider(fig[5,2], range=0:step:1, startvalue=sl_time_center_pref)
    gl3 = GridLayout(fig[6,3:4])
    bt_left_big_width = Button(gl3[1,1], label="<<")
    bt_left_small_width = Button(gl3[1,2], label="<")
    bt_right_small_width = Button(gl3[1,3], label=">")
    bt_right_big_width = Button(gl3[1,4], label=">>")
    Label(fig[6,2, Left()], "width")
    maxvalue = max_width_sec * fs[] / length(y[])
    sl_time_width = Slider(fig[6,2], range=0:step:maxvalue,
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

    mtspectrums = lift(y, to.active, cb_pval.checked, fs, tapers, iclip, nffts) do y, to, pval, fs, tapers, iclip, nffts
        if !to || pval
            mtspectrums = []
            for (nfft, taper) in zip(nffts, tapers)
                push!(mtspectrums, multitaper_spectrogram(iclip..., nfft; fs=fs, tapers=taper))
            end
            return mtspectrums
        else
            fill(Vector{MTSpectrum}(undef, 0), 0)
        end
    end

    Y_MTs = lift(mtspectrums, to.active) do mts, to
        if !to
            Y_MTs = []
            for mt in mts
                p = hcat((x.S for x in mt)...)
                f = mt[1].f
                t = (1:length(mt)) * mt[1].params.dt
                push!(Y_MTs, DSP.Periodograms.Spectrogram(p, f, t))
            end
            return Y_MTs
        else
            fill(DSP.Periodograms.Spectrogram(Matrix{Float64}(undef, 0, 0), 0:0., 0:0.), 0)
        end
    end
    Y_MT = @lift $(to.active) ? Matrix{RGB{N0f8}}(undef, 0, 0) : overlay($Y_MTs, x->20*log10.(power(x)), [0.01,0.99])

    Fs = lift(mtspectrums, cb_pval.checked) do mts, pval
        if pval
            Fs = []
            for mt in mts
                push!(Fs, hcat((x.Fpval for x in mt)...))
            end
            return Fs
        else
            fill(Matrix{Float64}(undef, 0, 0), 0)
        end
    end
    F = lift(Fs, cb_pval.checked, cb_sigonly.checked, thresh) do Fs, pval, sigonly, thresh
        if pval
            Y_colored = overlay(Fs, identity, (thresh, 0.99))
            idx = findall(isequal(RGBA{N0f8}(0.0, 0.0, 0.0, 1)), Y_colored)
            sigonly && (Y_colored .= RGBA{N0f8}(0.0, 0.0, 0.0, 1.0))
            Y_colored[idx] .= RGBA{N0f8}(1.0, 0.0, 1.0, 1.0)
            return Y_colored
        else
            Matrix{RGBA{N0f8}}(undef, 0, 0)
        end
    end

    alpha_power = @lift $(cb_power.checked) * 0.5 + !$(cb_pval.checked) * 0.5
    alpha_pval = @lift $(cb_pval.checked) * 0.5 + !$(cb_power.checked) * 0.5

    powers = lift(to.active, Y, itime, ifreq, Y_MT) do to, Y, itime, ifreq, Y_MT
        if to
            all(in.(extrema(itime), Ref(axes(Y,2)))) || return RGBA{N0f8}[1 0; 0 0]
            all(in.(extrema(ifreq), Ref(axes(Y,1)))) || return RGBA{N0f8}[1 0; 0 0]
            return Y'[itime,ifreq]
        else
            all(in.(extrema(ifreq), Ref(axes(Y_MT,1)))) || return RGBA{N0f8}[1 0; 0 0]
            return Y_MT'[1:itime.step:end, ifreq]
        end
    end

    freqs = @lift tuple($Y_freq[$ifreq][[1,end]] ./ hz2khz...)
    times = @lift tuple($Y_time[$itime][[1,end]]...)
    ax,hm = image(fig[3:4,2], times, freqs, powers;
                  interpolate=false, alpha=alpha_power, visible=cb_power.checked)

    ax.xlabel[] = "time (s)"
    ax.ylabel[] = "frequency (kHz)"
    onany(freqs, times) do f,t
        limits!(ax, t..., f...)
    end

    freqs_mt = @lift $(cb_pval.checked) ? $freqs : tuple(0.,0.)
    times_mt = @lift $(cb_pval.checked) ? $times : tuple(0.,0.)
    pvals = lift(cb_pval.checked, F, itime, ifreq) do pval, F, itime, ifreq
        if pval
            all(in.(extrema(ifreq), Ref(axes(F,1)))) || return RGBA{N0f8}[1 0; 0 0]
            F'[1:itime.step:end, ifreq]
        else
            Matrix{RGBA{N0f8}}(undef, 1, 1)
        end
    end
    cr = @lift ($thresh,1)
    hm_pvals = image!(times_mt, freqs_mt, pvals;
                      interpolate=false,
                      colormap=:grays, colorrange=cr, lowclip=(:fuchsia, 1),
                      alpha=alpha_pval,
                      visible=cb_pval.checked)

    y_clip = @lift view(y[], $iclip[1] : max(1, fld($iclip[2]-$iclip[1], display_size[2])) : $iclip[2])

    ax1, li1 = lines(fig[7,2], y_clip)
    ax1.xticklabelsvisible[] = ax1.yticklabelsvisible[] = false
    ax1.ylabel[] = "amplitude"
    on(yc->limits!(ax1, 1, length(yc), extrema(yc)...), y_clip)

    function _cumpower(p,d)
        x = dropdims(sum(p, dims=d), dims=d)
        @. red(x) + green(x) + blue(x)
    end

    cumpowers1 = @lift _cumpower($powers, 1)
    cumpowers1_freqs = @lift Point2f.(zip($cumpowers1, 1:length($cumpowers1)))

    ax2, li2 = lines(fig[3:4,3], cumpowers1_freqs)
    ax2.xticklabelsvisible[] = ax2.yticklabelsvisible[] = false
    ax2.xlabel[] = "power"
    onany(cp->limits!(ax2, extrema(cp)..., 1, length(cp)), cumpowers1)

    cumpowers2 = @lift _cumpower($powers, 2)

    ax3, li3 = lines(fig[2,2], cumpowers2)
    ax3.xticklabelsvisible[] = ax3.yticklabelsvisible[] = false
    ax3.ylabel[] = "power"
    onany(cp->limits!(ax3, 1, length(cp), extrema(cp)...), cumpowers2)

    colsize!(fig.layout, 2, Auto(8))
    colsize!(fig.layout, 3, Auto(1))
    rowsize!(fig.layout, 2, Auto(1))
    rowsize!(fig.layout, 3, Auto(4))

    onany(bt_play.clicks) do _
        yfilt = filtfilt(digitalfilter(Lowpass(fs_play/2/fs[]), Butterworth(4)),
                         y[][iclip[][1]:iclip[][2], 1])
        ydown = resample(yfilt, fs_play/fs[])
        wavplay(ydown, fs_play)
    end

    on(x->@set_preferences!("figsize"=>string(tuple(x.widths...))), fig.scene.viewport)
    on(x->@set_preferences!("isl_freq"=>string(x)), isl_freq.interval)
    on(x->@set_preferences!("sl_time_center"=>x), sl_time_center.value)
    on(x->@set_preferences!("sl_time_width"=>x), sl_time_width.value)
    on(x->@set_preferences!("m"=>x), m.selection)
    on(x->@set_preferences!("cb_pval"=>x), cb_pval.checked)
    on(x->@set_preferences!("cb_sigonly"=>x), cb_sigonly.checked)
    on(x->@set_preferences!("tb_thresh"=>x), tb_thresh.stored_string)
    on(x->@set_preferences!("cb_power"=>x), cb_power.checked)
    on(x->@set_preferences!("to"=>x), to.active)
    on(x->@set_preferences!("tb_nfft"=>x), tb_nfft.stored_string)

    notify(freqs)
    notify(cumpowers1)
    notify(cumpowers2)
    notify(y_clip)
    display(fig)
end

end # module SentimentAnalysis
