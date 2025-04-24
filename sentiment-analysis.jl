using WAV, DSP, GLMakie, Multitaper, Memoize, LRUCache

basepath = "/Volumes/karpova/TervoLab/data/Vocals/RatCity/cohort3/ch1"

hz2khz = 1000
fs_play = 48_000
nw, k = 4.0, 6
display_size = (3000,2000)
max_width_sec = 60

fig = Figure()

m = Menu(fig[1,1:3], options=filter(endswith(".wav"), readdir(basepath)))

y_fs_ = @lift wavread(joinpath(basepath, $(m.selection)))
y = @lift $(y_fs_)[1]
fs = @lift $(y_fs_)[2]

gl = GridLayout(fig[2:4,4])
Label(gl[1,1, Top()], "p-val")
cb_pval = Checkbox(gl[1,1], checked = false)
Label(gl[2,1,Top()], "threshold")
tb_thresh = Textbox(gl[2,1], stored_string="0.01", validator=Float64)
Label(gl[3,1, Top()], "power")
cb_power = Checkbox(gl[3,1], checked = true)
Label(gl[4,1, Right()], "Hanning\nSlepian")
to = Toggle(gl[4,1], active=true, orientation=:vertical)
Label(gl[5,1,Top()], "nfft")
tb_nfft = Textbox(gl[5,1], stored_string="512", validator=Int)
bt_play = Button(gl[6,1], label="play")

thresh = @lift parse(Float64, $(tb_thresh.stored_string))
nfft = @lift parse(Int, $(tb_nfft.stored_string))

Y = @lift spectrogram($y[:,1], $nfft; fs=$fs, window=hanning)

tapers = @lift dpss_tapers($nfft, nw, k, :tap)

@memoize LRU(maxsize=100_000) _multispec(idx, n, NW, K, dt, dpVec, Ftest) =
        multispec((@view y[][idx:idx+n-1]), NW=NW, K=K, dt=dt, dpVec=dpVec, Ftest=Ftest)
function multitaper_spectrogram(i1, i2, n; fs=1, nw=4.0, k=6, tapers=tapers)
    noverlap = div(n, 2)
    idxs = i1 : n-noverlap : i2+1-n+1
    mtspectrum = Vector{MTSpectrum}(undef, length(idxs))
    Threads.@threads :greedy for (i,idx) in enumerate(idxs)
        mtspectrum[i] = _multispec(idx, n, nw, k, 1/fs, tapers, true)
    end
    return mtspectrum
end

o_rfreq = @lift 1 : length(freq($Y))
o_rtime = @lift 1 : length(time($Y))
sl_freq = IntervalSlider(fig[3:4,1], range=o_rfreq, horizontal=false)
gl2 = GridLayout(fig[5,3])
bt_left_center = Button(gl2[1,1], label="<")
bt_right_center = Button(gl2[1,2], label=">")
Label(fig[5,2, Left()], "center")
sl_time_center = Slider(fig[5,2], range=o_rtime)
gl3 = GridLayout(fig[6,3])
bt_left_width = Button(gl3[1,1], label="<")
bt_right_width = Button(gl3[1,2], label=">")
Label(fig[6,2, Left()], "width")
maxvalue = Int(cld(max_width_sec*fs[], nfft[]/2))
sl_time_width = Slider(fig[6,2], range=1:maxvalue, startvalue=maxvalue)

on(_->set_close_to!(sl_time_center, sl_time_center.value[] - sl_time_width.value[] / 10),
   bt_left_center.clicks)
on(_->set_close_to!(sl_time_center, sl_time_center.value[] + sl_time_width.value[] / 10),
   bt_right_center.clicks)
on(_->set_close_to!(sl_time_width, sl_time_width.value[]*0.9),
   bt_left_width.clicks)
on(_->set_close_to!(sl_time_width, sl_time_width.value[]*1.1),
   bt_right_width.clicks)

ifreq = lift(x -> x[1] : max(1, fld(x[2]-x[1], display_size[2])) : x[2], sl_freq.interval)
itime = lift((c,w,Y) -> max(1,c-w) : max(1, fld(w, display_size[1])) : min(length(time(Y)),c+w),
             sl_time_center.value, sl_time_width.value, Y)

iclip = lift(y, nfft, itime) do y, nfft, itime
    noverlap = div(nfft, 2)
    i1 = 1 + itime[1] * (nfft - noverlap)
    i2 = nfft + itime[end] * (nfft - noverlap)
    (i1, i2)
end

mtspectrums = lift(y, to.active, cb_pval.checked, fs, tapers, iclip, nfft) do y, to, pval, fs, tapers, iclip, nfft
    if !to || pval
        multitaper_spectrogram(iclip..., nfft; fs=fs, tapers=tapers)
    else
        Vector{MTSpectrum}(undef, 0)
    end
end

Y_MT = lift(mtspectrums, to.active) do mts, to
    if !to
        p = hcat((x.S for x in mts)...)
        f = mts[1].f
        t = (1:length(mts)) * mts[1].params.dt
        DSP.Periodograms.Spectrogram(p, f, t)
    else
        DSP.Periodograms.Spectrogram(Matrix{Float64}(undef, 0, 0), 0:0., 0:0.)
    end
end

F = lift(mtspectrums, cb_pval.checked) do mts, pval
    if pval
        hcat((x.Fpval for x in mts)...)
    else
        Matrix{Float64}(undef, 0, 0)
    end
end

alpha_power = @lift $(cb_power.checked) * 0.5 + !$(cb_pval.checked) * 0.5
alpha_pval = @lift $(cb_pval.checked) * 0.5 + !$(cb_power.checked) * 0.5

powers = @lift 20*log10.($(to.active) ? power($Y)'[$itime,$ifreq] :
                                        power($Y_MT)'[1:$itime.step:end, $ifreq])
freqs = @lift freq($Y)[$ifreq] ./ hz2khz
times = @lift time($Y)[$itime]
ax,hm = heatmap(fig[3:4,2], times, freqs, powers; alpha=alpha_power, visible=cb_power.checked)

ax.xlabel[] = "time (s)"
ax.ylabel[] = "frequency (kHz)"
onany(freqs, times) do f,t
    limits!(ax, t[1], t[end], f[1], f[end])
end

freqs_mt = @lift $(cb_pval.checked) ? $freqs : Vector{Float32}(undef, 1)
times_mt = @lift $(cb_pval.checked) ? $times : Vector{Float32}(undef, 1)
pvals = @lift $(cb_pval.checked) ? ($F)'[1:$itime.step:end, $ifreq] : Matrix{Float64}(undef, 1, 1)
cr = @lift ($thresh,1)
hm_pvals = heatmap!(times_mt, freqs_mt, pvals;
                    colormap=:grays, colorrange=cr, lowclip=(:fuchsia, 1),
                    alpha=alpha_pval,
                    visible=cb_pval.checked)

y_clip = @lift view(y[], $iclip[1] : max(1, fld($iclip[2]-$iclip[1], display_size[2])) : $iclip[2])

ax1, li1 = lines(fig[7,2], y_clip)
ax1.xticklabelsvisible[] = ax1.yticklabelsvisible[] = false
ax1.ylabel[] = "amplitude"
on(yc->limits!(ax1, 1, length(yc), extrema(yc)...), y_clip)

_cumpower(p,d) = dropdims(sum(p, dims=d), dims=d)

cumpowers1 = @lift _cumpower($powers, 1)
cumpowers1_freqs = @lift Point2f.(zip($cumpowers1, $freqs))

ax2, li2 = lines(fig[3:4,3], cumpowers1_freqs)
ax2.xticklabelsvisible[] = ax2.yticklabelsvisible[] = false
ax2.xlabel[] = "power"
onany((f,cp)->limits!(ax2, extrema(cp)..., f[1], f[end]), freqs, cumpowers1)

cumpowers2 = @lift _cumpower($powers, 2)
times_cumpowers2 = @lift Point2f.(zip($times, $cumpowers2))

ax3, li3 = lines(fig[2,2], times_cumpowers2)
ax3.xticklabelsvisible[] = ax3.yticklabelsvisible[] = false
ax3.ylabel[] = "power"
onany((t,cp)->limits!(ax3, t[1], t[end], extrema(cp)...), times, cumpowers2)

colsize!(fig.layout, 2, Auto(8))
colsize!(fig.layout, 3, Auto(1))
rowsize!(fig.layout, 2, Auto(1))
rowsize!(fig.layout, 3, Auto(4))

onany(bt_play.clicks) do _
    overlap = nfft[] / 2
    t0 = round(Int, itime[][1] * overlap)
    t1 = round(Int, (1+itime[][end]) * overlap)
    yfilt = filtfilt(digitalfilter(Lowpass(fs_play/2/fs[]), Butterworth(4)), y[][t0:t1,1])
    ydown = resample(yfilt, fs_play/fs[])
    wavplay(ydown, fs_play)
end

notify(freqs)
display(fig)
