using WAV, DSP, GLMakie, Multitaper

basepath = "/Volumes/karpova/TervoLab/data/Vocals/RatCity/cohort3/ch1"

hz2khz = 1000
hm_res = 1000
fs_play = 48_000
nw, k = 4.0, 6

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

Y = @lift spectrogram($y[1:250_000*10,1], parse(Int, $(n_tb.stored_string)); fs=$fs, window=hanning)


tapers = @lift dpss_tapers(parse(Int, $(n_tb.stored_string)), nw, k, :tap)

function multitaper_spectrogram(s, n=div(length(s), 8), noverlap=div(n, 2); fs=1, nw=4.0, k=6)
    i0 = 1 : n-noverlap : length(s)-n+1
    mtspectrum = Vector{MTSpectrum}(undef, length(i0))
    Threads.@threads :greedy for (i,idx) in enumerate(i0)
        mtspectrum[i] = multispec(s[idx:idx+n-1], NW=nw, K=k, dt=1/fs, dpVec=tapers, Ftest=true)
    end
    return mtspectrum
end

mtspectrums = @lift multitaper_spectrogram($y[1:250_000*10,1], parse(Int, $(n_tb.stored_string)); fs=$fs)

Y_MT = lift(mtspectrums) do mts
    p = hcat((x.S for x in mts)...)
    f = mts[1].f
    t = (1:length(mts)) * mts[1].params.dt
    DSP.Periodograms.Spectrogram(p, f, t)
end

F = lift(mtspectrums) do mts
    hcat((x.Fpval for x in mts)...)
end

o_rfreq = @lift 0 : 1/(length(freq($Y))-1) : 1
o_rtime = @lift 0 : 1/(length(time($Y))-1) : 1

sl_freq = IntervalSlider(fig[3:4,1], range=o_rfreq, horizontal=false)
sl_time = IntervalSlider(fig[5,2], range=o_rtime)

ifreq = lift(sl_freq.interval, Y) do sl_freq, Y
    i0 = round(Int, 1 + sl_freq[1] * (length(freq(Y))-1))
    i1 = round(Int, 1 + sl_freq[2] * (length(freq(Y))-1))
    df = max(1, floor(Int, (i1-i0)/hm_res))
    i0:df:i1
end
itime = lift(sl_time.interval, Y) do sl_time, Y
    i0 = round(Int, 1 + sl_time[1] * (length(time(Y))-1))
    i1 = round(Int, 1 + sl_time[2] * (length(time(Y))-1))
    df = max(1, floor(Int, (i1-i0)/hm_res))
    i0:df:i1
end

hm_vis = @lift $(to.active) && $(cb_power.checked)
hm_mt_vis = @lift !$(to.active) && $(cb_power.checked)
alpha_power = @lift $(cb_power.checked) * 0.5 + !$(cb_pval.checked) * 0.5
alpha_pval = @lift $(cb_pval.checked) * 0.5 + !$(cb_power.checked) * 0.5

powers = @lift 20*log10.(power($Y)'[$itime,$ifreq])
freqs = @lift freq($Y)[$ifreq] ./ hz2khz
times = @lift time($Y)[$itime]
ax,hm = heatmap(fig[3:4,2], times, freqs, powers; alpha=alpha_power, visible=hm_vis)

ax.xlabel[] = "time (s)"
ax.ylabel[] = "frequency (kHz)"
onany(freqs, times) do f,t
    limits!(ax, t[1], t[end], f[1], f[end])
end

powers_mt = @lift 20*log10.(power($Y_MT)'[$itime,$ifreq])
hm_mt = heatmap!(times, freqs, powers_mt; alpha=alpha_power, visible=hm_mt_vis)

pvals = @lift ($F)'[$itime,$ifreq]
hm_pvals = heatmap!(times, freqs, pvals;
                    colormap=:grays, colorrange=(0.01,1), lowclip=(:fuchsia, 1),
                    alpha=alpha_pval,
                    visible=cb_pval.checked)

_cumpower(p,d) = dropdims(sum(p, dims=d), dims=d)

cumpowers1 = Observable(_cumpower(powers[],1))
on(p->cumpowers1.val=_cumpower(p,1), powers)
on(_->notify(cumpowers1), sl_time.interval)

ax2, li = lines(fig[3:4,3], cumpowers1, freqs)
ax2.width[]=100
ax2.xticklabelsvisible[]=ax2.yticklabelsvisible[]=false
onany((f,cp)->limits!(ax2, extrema(cp)..., f[1], f[end]), freqs, cumpowers1)

cumpowers2 = Observable(_cumpower(powers[],2))
on(p->cumpowers2.val=_cumpower(p,2), powers)
on(_->notify(cumpowers2), sl_freq.interval)

ax3, li = lines(fig[2,2], times, cumpowers2)
ax3.height[]=100
ax3.xticklabelsvisible[]=ax3.yticklabelsvisible[]=false
onany((t,cp)->limits!(ax3, t[1], t[end], extrema(cp)...), times, cumpowers2)

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
