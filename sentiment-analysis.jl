using WAV, DSP, GLMakie, Multitaper, Memoize, LRUCache

basepath = "/Volumes/karpova/TervoLab/data/Vocals/RatCity/cohort3/ch1"

hz2khz = 1000
fs_play = 48_000
nw, k = 4.0, 6
display_size = (3000,2000)

fig = Figure()

m = Menu(fig[1,1:3], options=filter(endswith(".wav"), readdir(basepath)))

y_fs_ = @lift wavread(joinpath(basepath, $(m.selection)))
y = @lift $(y_fs_)[1]
fs = @lift $(y_fs_)[2]

gl = GridLayout(fig[2:4,4])
Label(gl[1,1, Top()], "p-val")
cb_pval = Checkbox(gl[1,1], checked = false)
Label(gl[2,1, Top()], "power")
cb_power = Checkbox(gl[2,1], checked = true)
Label(gl[3,1, Right()], "Hanning\nSlepian")
to = Toggle(gl[3,1], active=true, orientation=:vertical)
Label(gl[4,1,Top()], "nfft")
n_tb = Textbox(gl[4,1], stored_string="512", validator=s->all(isdigit(c) for c in s))
play_bt = Button(gl[5,1], label="play")

Y = @lift spectrogram($y[:,1], parse(Int, $(n_tb.stored_string)); fs=$fs, window=hanning)


tapers = @lift dpss_tapers(parse(Int, $(n_tb.stored_string)), nw, k, :tap)

@memoize LRU(maxsize=100_000) _multispec(idx, n, NW, K, dt, dpVec, Ftest) =
        multispec((@view y[][idx:idx+n-1]), NW=NW, K=K, dt=dt, dpVec=dpVec, Ftest=Ftest)
function multitaper_spectrogram(i1, i2, n, noverlap=div(n, 2); fs=1, nw=4.0, k=6, tapers=tapers)
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
sl_time = IntervalSlider(fig[5,2], range=o_rtime)

ifreq = lift(x -> x[1] : max(1, fld(x[2]-x[1], display_size[2])) : x[2], sl_freq.interval)
itime = lift(x -> x[1] : max(1, fld(x[2]-x[1], display_size[1])) : x[2], sl_time.interval)

mtspectrums = lift(y, n_tb.stored_string, to.active, cb_pval.checked, itime, fs, tapers) do y, n_str, to, pval, itime, fs, tapers
    if !to || pval
        n = parse(Int, n_str)
        i1 = 1 + itime[1] * (n - div(n,2))
        i2 = n + itime[end] * (n - div(n,2))
        multitaper_spectrogram(i1, i2, n; fs=fs, tapers=tapers)
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
hm_pvals = heatmap!(times_mt, freqs_mt, pvals;
                    colormap=:grays, colorrange=(0.01,1), lowclip=(:fuchsia, 1),
                    alpha=alpha_pval,
                    visible=cb_pval.checked)

_cumpower(p,d) = dropdims(sum(p, dims=d), dims=d)

cumpowers1 = Observable(_cumpower(powers[],1))
on(p->cumpowers1.val=_cumpower(p,1), powers)
on(_->notify(cumpowers1), sl_time.interval)

ax2, li = lines(fig[3:4,3], cumpowers1, freqs)
ax2.xticklabelsvisible[]=ax2.yticklabelsvisible[]=false
onany((f,cp)->limits!(ax2, extrema(cp)..., f[1], f[end]), freqs, cumpowers1)

cumpowers2 = Observable(_cumpower(powers[],2))
on(p->cumpowers2.val=_cumpower(p,2), powers)
on(_->notify(cumpowers2), sl_freq.interval)

ax3, li = lines(fig[2,2], times, cumpowers2)
ax3.xticklabelsvisible[]=ax3.yticklabelsvisible[]=false
onany((t,cp)->limits!(ax3, t[1], t[end], extrema(cp)...), times, cumpowers2)

colsize!(fig.layout, 2, Auto(8))
colsize!(fig.layout, 3, Auto(1))
rowsize!(fig.layout, 2, Auto(1))
rowsize!(fig.layout, 3, Auto(4))

onany(play_bt.clicks, y, fs) do _, y, fs
    nfft = parse(Int, n_tb.stored_string[])
    overlap = nfft / 2
    t0 = round(Int, itime[][1] * overlap)
    t1 = round(Int, (1+itime[][end]) * overlap)
    yfilt = filtfilt(digitalfilter(Lowpass(fs_play/2/fs), Butterworth(4)), y[t0:t1,1])
    ydown = resample(yfilt, fs_play/fs)
    wavplay(ydown, fs_play)
end

notify(freqs)
display(fig)
