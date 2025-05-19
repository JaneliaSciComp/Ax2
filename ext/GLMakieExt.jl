module GLMakieExt

using Ax2, GLMakie, DelimitedFiles, Preferences, Colors, FixedPointNumbers, Statistics

include("block-tooltips.jl")

@kwdef struct Widgets
    fig; me_wav; me_jump; cb_mistakes;
    cb_power; to_window; tb_nfft; cb_ftest; tb_nwk; tb_pval; cb_sigonly;
    cb_morphclose; tb_strelclose; cb_morphopen; tb_strelopen; tb_minpix;
    isl_freq;
    bt_left_big_center; bt_left_small_center; bt_right_small_center; bt_right_big_center;
    bt_left_big_width; bt_left_small_width; bt_right_small_width; bt_right_big_width;
    sl_time_center; sl_time_width;
    lb_status; bt_play; bt_csv; bt_hdf;
    ax; hm; hm_pvals; l_hit; l_miss; l_fa; ax1; li1; ax2; li2; ax3; li3;
end

@kwdef struct Observables
    y; fs; hits; misses; false_alarms;
    nffts; noverlaps; offset; nw; k; pval; strelclose; strelopen; minpix;
    Ys; Y; Y_freq; Y_time; coarse2fine;
    ifreq; itime; iclip; mtspectrums; Y_MTs; Y_MT; Fs; F;
    alpha_power; alpha_pval; powers; freqs; times; freqs_mt; times_mt; pvals; cr;
    obs_hit; obs_miss; obs_fa;
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
    me_wav = missing,
    me_jump = "time",
    cb_mistakes = true,
    cb_tooltips = false,
    cb_power = true,
    to_window = true,
    tb_nfft = "512",
    cb_ftest = false,
    tb_nwk = "4,6",
    tb_pval = "0.01",
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
    me_wav_pref = @load_preference("me_wav", pref_defaults.me_wav)
    me_jump_pref = @load_preference("me_jump", pref_defaults.me_jump)
    cb_mistakes_pref = @load_preference("cb_mistakes", pref_defaults.cb_mistakes)
    cb_tooltips_pref = @load_preference("cb_tooltips", pref_defaults.cb_tooltips)
    cb_power_pref = @load_preference("cb_power", pref_defaults.cb_power)
    to_window_pref = @load_preference("to_window", pref_defaults.to_window)
    tb_nfft_pref = @load_preference("tb_nfft", pref_defaults.tb_nfft)
    cb_ftest_pref = @load_preference("cb_ftest", pref_defaults.cb_ftest)
    tb_nwk_pref = @load_preference("tb_nwk", pref_defaults.tb_nwk)
    tb_pval_pref = @load_preference("tb_pval", pref_defaults.tb_pval)
    cb_sigonly_pref = @load_preference("cb_sigonly", pref_defaults.cb_sigonly)
    cb_morphclose_pref = @load_preference("cb_morphclose", pref_defaults.cb_morphclose)
    tb_strelclose_pref = @load_preference("tb_strelclose", pref_defaults.tb_strelclose)
    cb_morphopen_pref = @load_preference("cb_morphopen", pref_defaults.cb_morphopen)
    tb_strelopen_pref = @load_preference("tb_strelopen", pref_defaults.tb_strelopen)
    tb_minpix_pref = @load_preference("tb_minpix", pref_defaults.tb_minpix)

    fig = Figure(size=figsize_pref)

    me_wav = Menu(fig[1,1:3], options=wavfiles, default=coalesce(me_wav_pref, wavfiles[1]))

    y_fs_ = @lift load_recording(joinpath(datapath, $(me_wav.selection)))
    y = @lift $(y_fs_)[1]
    fs = @lift Float64($(y_fs_)[2])

    hits = @lift begin
        fn = joinpath(datapath, string(splitext($(me_wav.selection))[1], "-partial-hits.csv"))
        isfile(fn) ? readdlm(fn, ',', header=true)[1] : missing
    end

    misses = @lift begin
        fn = joinpath(datapath, string(splitext($(me_wav.selection))[1], "-complete-misses.csv"))
        isfile(fn) ? readdlm(fn, ',', header=true)[1] : missing
    end

    false_alarms = @lift begin
        fn = joinpath(datapath, string(splitext($(me_wav.selection))[1], "-complete-false-alarms.csv"))
        isfile(fn) ? readdlm(fn, ',', header=true)[1] : missing
    end

    gl_tt = GridLayout(fig[2,3])
    Label(gl_tt[1,1, Bottom()], "tooltips", tellheight=false, tellwidth=false)
    cb_tooltips = Checkbox(gl_tt[2,1, Top()], checked = cb_tooltips_pref,
                           tellheight=false, tellwidth=false)

    gl_fft = GridLayout(fig[1:3,4])
    Label(gl_fft[1,1, Top()], "power")
    cb_power = Checkbox(gl_fft[1,1], checked = cb_power_pref)
    Label(gl_fft[2,1, Top()], "Hanning")
    Label(gl_fft[2,1, Bottom()], "Slepian")
    to_window = Toggle(gl_fft[2,1], active=to_window_pref, orientation=:vertical)
    Label(gl_fft[3,1,Top()], "nfft")
    tb_nfft = Textbox(gl_fft[3,1], stored_string=tb_nfft_pref,
                      validator = s -> all(isdigit(c) || c==',' for c in s))

    Label(gl_fft[4,1, Top()], "F-test")
    cb_ftest = Checkbox(gl_fft[4,1], checked = cb_ftest_pref)
    Label(gl_fft[5,1,Top()], "NW,K")
    tb_nwk = Textbox(gl_fft[5,1], stored_string=tb_nwk_pref,
                     validator = s -> all(isdigit(c) || in(c,".,") for c in s))
    Label(gl_fft[6,1,Top()], "p-val")
    tb_pval = Textbox(gl_fft[6,1], stored_string=tb_pval_pref, validator=Float64)
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
    offset = Observable(0)
    nw = @lift parse(Float64, split($(tb_nwk.stored_string), ',')[1])
    k = @lift parse(Int, split($(tb_nwk.stored_string), ',')[2])
    pval = @lift parse(Float64, $(tb_pval.stored_string))
    strelclose = @lift make_strel(tuple(parse.(Int, split($(tb_strelclose.stored_string), 'x'))...))
    strelopen = @lift make_strel(tuple(parse.(Int, split($(tb_strelopen.stored_string), 'x'))...))
    minpix = @lift parse(Int, $(tb_minpix.stored_string))

    Ys = @lift calculate_hanning_spectrograms($y, $nffts, $noverlaps, $offset, $fs)
    Y = @lift overlay($Ys, dB)

    Y_freq = @lift freq($Ys[argmax($nffts)])
    Y_time = @lift time($Ys[argmin($nffts)])
    coarse2fine = @lift div(length($Y_time), length(time($Ys[argmax($nffts)])))

    isl_freq = IntervalSlider(fig[3,1], range=0:0.01:1, horizontal=false,
                              startvalues = coalesce(isl_freq_pref, tuple(0, 1)))
    gl_pan = GridLayout(fig[4,3:5], halign=:left)
    bt_left_big_center = Button(gl_pan[1,1], label="<<")
    bt_left_small_center = Button(gl_pan[1,2], label="<")
    bt_right_small_center = Button(gl_pan[1,3], label=">")
    bt_right_big_center = Button(gl_pan[1,4], label=">>")
    Label(fig[4,2, Left()], "center")
    step = (nffts[][1]-noverlaps[][1]) / length(y[])
    sl_time_center = Slider(fig[4,2], range=0:step:1, startvalue=sl_time_center_pref)

    me_jump = Menu(gl_pan[1,5], options = ["time", "hits", "misses", "FAs"],
                   default=me_jump_pref, width=70)

    gl_zoom = GridLayout(fig[5,3:5], halign=:left)
    bt_left_big_width = Button(gl_zoom[1,1], label="<<")
    bt_left_small_width = Button(gl_zoom[1,2], label="<")
    bt_right_small_width = Button(gl_zoom[1,3], label=">")
    bt_right_big_width = Button(gl_zoom[1,4], label=">>")
    Label(fig[5,2, Left()], "width")
    maxvalue = max_width_sec * fs[] / length(y[])
    sl_time_width = Slider(fig[5,2], range=0:step:maxvalue,
                           startvalue = coalesce(sl_time_width_pref, maxvalue))

    Label(gl_zoom[1,5, Right()], "show hits &\nmistakes")
    cb_mistakes = Checkbox(gl_zoom[1,5], checked = cb_mistakes_pref)

    function jump(data, fun1, fun2, skip)
        t = mean(Y_time[][itime[][[1,end]]])
        i = fun2(x -> fun1(x, t), data[][:,1])
        isnothing(i) && return
        i = clamp(i+skip, 1, size(data[],1))
        data[][i,1] * fs[] / length(y[])
    end

    on(bt_left_big_center.clicks) do _
        if me_jump.selection[]=="time"
            set_close_to!(sl_time_center, sl_time_center.value[] - sl_time_width.value[] / 2)
        elseif me_jump.selection[]=="hits"
            c = jump(hits, <, findlast, -9)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="misses"
            c = jump(misses, <, findlast, -9)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="FAs"
            c = jump(false_alarms, <, findlast, -9)
            isnothing(c) || set_close_to!(sl_time_center, c)
        end
    end
    on(bt_left_small_center.clicks) do _
        if me_jump.selection[]=="time"
            set_close_to!(sl_time_center, sl_time_center.value[] - sl_time_width.value[] / 10)
        elseif me_jump.selection[]=="hits"
            c = jump(hits, <, findlast, 0)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="misses"
            c = jump(misses, <, findlast, 0)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="FAs"
            c = jump(false_alarms, <, findlast, 0)
            isnothing(c) || set_close_to!(sl_time_center, c)
        end
    end
    on(bt_right_small_center.clicks) do _
        if me_jump.selection[]=="time"
            set_close_to!(sl_time_center, sl_time_center.value[] + sl_time_width.value[] / 10)
        elseif me_jump.selection[]=="hits"
            c = jump(hits, >, findfirst, 1)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="misses"
            c = jump(misses, >, findfirst, 1)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="FAs"
            c = jump(false_alarms, >, findfirst, 1)
            isnothing(c) || set_close_to!(sl_time_center, c)
        end
    end
    on(bt_right_big_center.clicks) do _
        if me_jump.selection[]=="time"
            set_close_to!(sl_time_center, sl_time_center.value[] + sl_time_width.value[] / 2)
        elseif me_jump.selection[]=="hits"
            c = jump(hits, >, findfirst, 10)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="misses"
            c = jump(misses, >, findfirst, 10)
            isnothing(c) || set_close_to!(sl_time_center, c)
        elseif me_jump.selection[]=="FAs"
            c = jump(false_alarms, >, findfirst, 10)
            isnothing(c) || set_close_to!(sl_time_center, c)
        end
    end

    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*0.5),
       bt_left_big_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*0.9),
       bt_left_small_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*1.1),
       bt_right_small_width.clicks)
    on(_->set_close_to!(sl_time_width, sl_time_width.value[]*1.5),
       bt_right_big_width.clicks)

    lb_status = Label(fig[7,1:4], " ")

    gl_out = GridLayout(fig[6,3:5], tellheight=false)

    bt_play = Button(gl_out[1,1], label="play")
    on(_->play(y[], iclip[], fs[]), bt_play.clicks)

    bt_csv = Button(gl_out[1,2], label="CSV")
    on(bt_csv.clicks) do _
        if !cb_ftest.checked[] || !cb_sigonly.checked[]
            lb_status.text[] = "F-test and sig. only must both be checked to output CSV"
            return
        end
        filename = joinpath(datapath, string(me_wav.selection[], '-', iclip[][1], '-', iclip[][2], ".csv"))
        save_csv(filename, F[], Y_freq[], ifreq[], Y_time[], itime[])
        lb_status.text[] = "CSV saved to $filename"
    end

    bt_hdf = Button(gl_out[1,3], label="HDF")
    on(bt_hdf.clicks) do _
        if !cb_ftest.checked[] || !cb_sigonly.checked[]
            lb_status.text[] = "F-test and sig. only must both be checked to output HDF"
            return
        end
        filename = joinpath(datapath, string(me_wav.selection[], '-', iclip[][1], '-', iclip[][2], ".hdf"))
        save_hdf(filename, F[], Y_freq[], ifreq[], Y_time[], itime[])
        lb_status.text[] = "HDF saved to $filename"
    end

    # indices into Y
    ifreq = lift(isl_freq.interval, Y_freq) do x, Y_freq
        start = 1 + round(Int, (length(Y_freq)-1) * x[1])
        stop = 1 + round(Int, (length(Y_freq)-1) * x[2])
        step = max(1, fld(stop-start+1, display_size[2]))
        start:step:stop
    end
    itime = @lift begin
        c, w = $(sl_time_center.value), $(sl_time_width.value)
        frac2fft(x) = round(Int, x*length($Y_time) / $coarse2fine) * $coarse2fine
        start = max(1, frac2fft(c-w))
        step = max(1, Int(fld(frac2fft(w), display_size[1])))
        stop = min(length($Y_time), frac2fft(c+w))
        start:step:stop
    end

    # indices into y
    iclip = @lift begin
        (round(Int, Y_time[][$itime[1]]*fs[] - minimum(noverlaps[]) + 1 + $offset),
         round(Int, Y_time[][$itime[end]]*fs[] + minimum(noverlaps[]) + $offset))
    end

    mtspectrums = @lift begin
        if !$(to_window.active) || $(cb_ftest.checked)
            calculate_multitaper_spectrograms($y, $nffts, $noverlaps, $nw, $k, $fs, $iclip)
        else
            fill(Vector{Periodograms.PeriodogramF}(undef, 0), 0)
        end
    end

    Y_MTs = @lift begin
        if !$(to_window.active)
            coalesce_multitaper_power($mtspectrums)
        else
            fill(Periodograms.Spectrogram(Matrix{Float64}(undef, 0, 0), 0:0., 0:0.), 0)
        end
    end
    Y_MT = @lift $(to_window.active) ? Array{Float32}(undef, 0, 0, 0) : overlay($Y_MTs, dB)

    Fs = @lift begin
        if $(cb_ftest.checked)
            coalesce_multitaper_ftest($mtspectrums)
        else
            fill(Matrix{Float64}(undef, 0, 0), 0)
        end
    end
    F = @lift begin
        if $(cb_ftest.checked)
            refine_ftest($Fs, $pval, $(cb_sigonly.checked),
                           $(cb_morphclose.checked), $strelclose,
                           $(cb_morphopen.checked), $strelopen,
                           $minpix)
        else
            Matrix{RGBA{N0f8}}(undef, 0, 0)
        end
    end

    alpha_power = @lift $(cb_power.checked) * 0.5 + !$(cb_ftest.checked) * 0.5
    alpha_pval = @lift $(cb_ftest.checked) * 0.5 + !$(cb_power.checked) * 0.5

    powers = @lift begin
        if $(to_window.active)
            all(in.(extrema($itime), Ref(axes($Y,3)))) || return RGBA{N0f8}[1 0; 0 0]
            all(in.(extrema($ifreq), Ref(axes($Y,2)))) || return RGBA{N0f8}[1 0; 0 0]
            Y_scratch = $Y[:,$ifreq,$itime]
        else
            all(in.(extrema($ifreq), Ref(axes($Y_MT,2)))) || return RGBA{N0f8}[1 0; 0 0]
            Y_scratch = $Y_MT[:, $ifreq, 1:$itime.step:end]
        end
        scale_and_color(Y_scratch)
    end

    freqs = @lift tuple($Y_freq[$ifreq[[1,end]]] ./ hz2khz...)
    times = @lift tuple($Y_time[$itime[[1,end]]]...)
    ax,hm = image(fig[3,2], times, freqs, powers;
                  interpolate=false, alpha=alpha_power, visible=cb_power.checked,
                  inspector_label = (pl,i,pos)->string(
                          "time = ", pos[1], " sec\n",
                          "freq = ", pos[2], " kHz\n",
                          "power = ", red(pos[3]).i+0, ',', green(pos[3]).i+0, ',', blue(pos[3]).i+0))

    ax.xlabel[] = "time (s)"
    ax.ylabel[] = "frequency (kHz)"
    onany(freqs, times) do f,t
        limits!(ax, t..., f...)
    end

    pvals = @lift begin
        if $(cb_ftest.checked)
            all(in.(extrema($ifreq), Ref(axes($F,1)))) || return RGBA{N0f8}[1 0; 0 0]
            $F'[1:$itime.step:end, $ifreq]
        else
            Matrix{RGBA{N0f8}}(undef, 1, 1)
        end
    end
    freqs_mt = @lift $(cb_ftest.checked) ? $freqs : tuple(0.,0.)
    times_mt = @lift $(cb_ftest.checked) ? tuple($Y_time[$itime[[1,size($pvals,1)]]]...) : tuple(0.,0.)
    cr = @lift ($pval,1)
    hm_pvals = image!(times_mt, freqs_mt, pvals;
                      interpolate=false,
                      colormap=:grays, colorrange=cr, lowclip=(:fuchsia, 1),
                      alpha=alpha_pval,
                      visible=cb_ftest.checked,
                      inspector_label = (pl,i,pos)->string(
                              "time = ", pos[1], " sec\n",
                              "freq = ", pos[2], " kHz\n",
                              "power = ", red(pos[3]).i+0, ',', green(pos[3]).i+0, ',', blue(pos[3]).i+0))

    obs_hit = @lift ismissing($hits) ? Point2f[(0, 0)] :
            [Rect(r[1], r[3]./hz2khz, r[2]-r[1], (r[4]-r[3])./hz2khz) for r in eachrow($hits)]
    l_hit = poly!(obs_hit, color = Cycled(2), visible=cb_mistakes.checked)
    obs_miss = @lift ismissing($misses) ? Point2f[(0, 0)] :
            [Rect(r[1], r[3]./hz2khz, r[2]-r[1], (r[4]-r[3])./hz2khz) for r in eachrow($misses)]
    l_miss = poly!(obs_miss, color = Cycled(3), visible=cb_mistakes.checked)
    obs_fa = @lift ismissing($false_alarms) ? Point2f[(0, 0)] :
            [Rect(r[1], r[3]./hz2khz, r[2]-r[1], (r[4]-r[3])./hz2khz) for r in eachrow($false_alarms)]
    l_fa = poly!(obs_fa, color = Cycled(4), visible=cb_mistakes.checked)

    iclip_subsampled = @lift $iclip[1] : max(1, fld($iclip[2]-$iclip[1], display_size[2])) : $iclip[2]
    y_clip = @lift view(y[], $iclip_subsampled)
    times_yclip = @lift Point2.(zip($iclip_subsampled ./ $fs, $y_clip))

    ax1, li1 = lines(fig[6,2], times_yclip,
                     inspector_label = (pl,i,pos)->string("time = ", pos[1], " sec\n",
                                                          "amplitude = ", pos[2], " V"))
    ax1.xticklabelsvisible[] = ax1.yticklabelsvisible[] = false
    ax1.ylabel[] = "amplitude"
    onany((yc,ics,fs)->limits!(ax1, ics[1]/fs, ics[end]/fs, extrema(yc)...),
          y_clip, iclip_subsampled, fs)

    cumpowers1 = @lift cumpower($powers, 1)
    cumpowers1_freqs = @lift Point2.(zip($cumpowers1, $Y_freq[$ifreq]))

    ax2, li2 = lines(fig[3,3], cumpowers1_freqs,
                     inspector_label = (pl,i,pos)->string("freq = ", pos[2], " kHz\n",
                                                          "power = ", pos[1], " dB"))
    ax2.xticklabelsvisible[] = ax2.yticklabelsvisible[] = false
    ax2.xlabel[] = "power"
    onany((cp,Yf,i)->limits!(ax2, extrema(cp)..., Yf[i[1]], Yf[i[end]]),
          cumpowers1, Y_freq, ifreq)

    cumpowers2 = @lift cumpower($powers, 2)
    times_cumpowers2 = @lift Point2.(zip($Y_time[$itime], $cumpowers2))

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

    tooltip!(me_wav, "choose a recording",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(cb_power, "calculate and display the spectrogram",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(to_window, "toggle between a normal spectrogram and a multitaper one",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_nfft,
             """
             short time windows can resolve quickly varying signals better
             long windows have better spectral resolution
             """,
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(cb_ftest, "calculate and display the multitaper F-test",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_nwk,
             """
             K is # of tapers
             NW is the spectral bandwidth
             K should be less than 2NW-1
             """,
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_pval, "the threshold at which to consider the F-test significant",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(cb_sigonly, "display the insignificant pixels in all black",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(cb_morphopen, "fill in small gaps between significant pixels",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_strelopen, "the height and width of the opening structuring element",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(cb_morphclose, "remove isolated significant pixels",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_strelclose, "the height and width of the closing structuring element",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(tb_minpix, "cull connected components with fewer than this many pixels",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(isl_freq, "zoom in on the frequency axis",
             placement = :right, enabled = cb_tooltips.checked)
    tooltip!(sl_time_center, "pan on the time axis",
             placement = :below, enabled = cb_tooltips.checked)
    tooltip!(sl_time_width, "zoom in on the time axis",
             placement = :below, enabled = cb_tooltips.checked)

    tooltip!(bt_left_big_center, "pan left by half",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_left_small_center, "pan left by a tenth",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_right_small_center, "pan right by a tenth",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_right_big_center, "pan right by half",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(bt_left_big_width, "zoom in by half",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_left_small_width, "zoom in pan left by a tenth",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_right_small_width, "zoom out by a tenth",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_right_big_width, "zoom out by half",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(bt_play, "listen to the displayed sound",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_csv, "save the displayed vocalizations to a CSV file",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(bt_hdf, "save the displayed vocalizations to an HDF file",
             placement = :left, enabled = cb_tooltips.checked)

    tooltip!(me_jump, "pan in time or jump to nearby hits & mistakes",
             placement = :left, enabled = cb_tooltips.checked)
    tooltip!(cb_mistakes, "show or hide vertical bands indicating hits & mistakes",
             placement = :left, enabled = cb_tooltips.checked)

    for cb in (cb_tooltips, cb_power, cb_ftest, cb_sigonly, cb_morphclose, cb_morphopen, cb_mistakes)
        foreach(x->x.inspectable[]=false, cb.blockscene.plots)
    end
    DataInspector(enabled=cb_tooltips.checked)

    on(x->@set_preferences!("figsize"=>string(tuple(x.widths...))), fig.scene.viewport)
    on(x->@set_preferences!("isl_freq"=>string(x)), isl_freq.interval)
    on(x->@set_preferences!("sl_time_center"=>x), sl_time_center.value)
    on(x->@set_preferences!("sl_time_width"=>x), sl_time_width.value)
    on(x->@set_preferences!("me_wav"=>x), me_wav.selection)
    on(x->@set_preferences!("me_jump"=>x), me_jump.selection)
    on(x->@set_preferences!("cb_mistakes"=>x), cb_mistakes.checked)
    on(x->@set_preferences!("cb_tooltips"=>x), cb_tooltips.checked)
    on(x->@set_preferences!("cb_power"=>x), cb_power.checked)
    on(x->@set_preferences!("to_window"=>x), to_window.active)
    on(x->@set_preferences!("tb_nfft"=>x), tb_nfft.stored_string)
    on(x->@set_preferences!("cb_ftest"=>x), cb_ftest.checked)
    on(x->@set_preferences!("tb_nwk"=>x), tb_nwk.stored_string)
    on(x->@set_preferences!("tb_pval"=>x), tb_pval.stored_string)
    on(x->@set_preferences!("cb_sigonly"=>x), cb_sigonly.checked)
    on(x->@set_preferences!("cb_morphclose"=>x), cb_morphclose.checked)
    on(x->@set_preferences!("tb_strelclose"=>x), tb_strelclose.stored_string)
    on(x->@set_preferences!("cb_morphopen"=>x), cb_morphopen.checked)
    on(x->@set_preferences!("tb_strelopen"=>x), tb_strelopen.stored_string)
    on(x->@set_preferences!("tb_minpix"=>x), tb_minpix.stored_string)

    widgets = Widgets(
        fig, me_wav, me_jump, cb_mistakes,
        cb_power, to_window, tb_nfft, cb_ftest, tb_nwk, tb_pval, cb_sigonly,
        cb_morphclose, tb_strelclose, cb_morphopen, tb_strelopen, tb_minpix,
        isl_freq,
        bt_left_big_center, bt_left_small_center, bt_right_small_center, bt_right_big_center,
        bt_left_big_width, bt_left_small_width, bt_right_small_width, bt_right_big_width,
        sl_time_center, sl_time_width,
        lb_status, bt_play, bt_csv, bt_hdf,
        ax, hm, hm_pvals, l_hit, l_miss, l_fa, ax1, li1, ax2, li2, ax3, li3
        )

    observables = Observables(
        y, fs, hits, misses, false_alarms,
        nffts, noverlaps, offset, nw, k, pval, strelclose, strelopen, minpix,
        Ys, Y, Y_freq, Y_time, coarse2fine,
        ifreq, itime, iclip, mtspectrums, Y_MTs, Y_MT, Fs, F,
        alpha_power, alpha_pval, powers, freqs, times, freqs_mt, times_mt, pvals, cr,
        obs_hit, obs_miss, obs_fa,
        iclip_subsampled, y_clip, times_yclip,
        cumpowers1, cumpowers1_freqs, cumpowers2, times_cumpowers2,
        )

    return widgets, observables
end

function Ax2.app(_datapath)
    global datapath, wavfiles
    datapath = _datapath
    wavfiles = filter(endswith(".wav"), readdir(_datapath))

    widgets, observables = init()
    notify(observables.freqs)
    notify(observables.cumpowers1)
    notify(observables.cumpowers2)
    notify(observables.y_clip)
    notify(observables.nffts)
    display(widgets.fig)
    return widgets, observables
end

end # module GLMakieExt
