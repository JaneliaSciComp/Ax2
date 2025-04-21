using WAV, DSP, GLMakie

#basepath = "/Users/arthurb/projects/tervo"
basepath = "/Volumes/karpova/TervoLab/data/Vocals/RatCity/cohort3/ch1"

#wavfile = "T2025-02-25_20-01-59_0000001_rat885.wav"
#wavfile = "T2025-03-02_16-19-23_0000003_rat885_eat_tickling.wav"
#wavfile = "T2025-03-02_16-23-15_0000004_rat886_eat_tickling.wav"
wavfile = "T2025-03-11_10-55-58_0000001.wav"

hz2khz = 1000
hm_res = 1000
fs_play = 48_000

y,fs,_ = wavread(joinpath(basepath,wavfile))

fig = Figure()

Label(fig[2,1,Top()], "nfft")
n_tb = Textbox(fig[2,1], stored_string="512", validator=s->all(isdigit(c) for c in s))

Y = @lift spectrogram(y[:,1], parse(Int, $(n_tb.stored_string)); fs=fs, window=hanning)

rfreq_o = @lift 0 : 1/(length(freq($Y))-1) : 1
rtime_o = @lift 0 : 1/(length(time($Y))-1) : 1

freq_sl = IntervalSlider(fig[1,1], range=rfreq_o, horizontal=false)
time_sl = IntervalSlider(fig[2,2], range=rtime_o)

ifreq = lift(freq_sl.interval, Y) do freq_sl, Y
    i0 = round(Int, 1 + freq_sl[1] * (length(freq(Y))-1))
    i1 = round(Int, 1 + freq_sl[2] * (length(freq(Y))-1))
    df = max(1, floor(Int, (i1-i0)/hm_res))
    i0:df:i1
end
itime = lift(time_sl.interval, Y) do time_sl, Y
    i0 = round(Int, 1 + time_sl[1] * (length(time(Y))-1))
    i1 = round(Int, 1 + time_sl[2] * (length(time(Y))-1))
    df = max(1, floor(Int, (i1-i0)/hm_res))
    i0:df:i1
end

powers = @lift 20*log10.(power($Y)'[$itime,$ifreq])
freqs = @lift freq($Y)[$ifreq] ./ hz2khz
times = @lift time($Y)[$itime]

ax,hm = heatmap(fig[1,2], times, freqs, powers)
ax.title[] = wavfile
ax.xlabel[] = "time (s)"
ax.ylabel[] = "frequency (kHz)"
onany(freqs, times) do f,t
    limits!(ax, t[1], t[end], f[1], f[end])
end

_cumpower(p) = dropdims(sum(p, dims=1), dims=1)
cumpowers = Observable(_cumpower(powers[]))
on(p->cumpowers.val=_cumpower(p), powers)
on(_->notify(cumpowers), time_sl.interval)

ax2, li = lines(fig[1,3], cumpowers, freqs)
ax2.width[]=100
ax2.xticklabelsvisible[]=false
onany((f,cp)->limits!(ax2, extrema(cp)..., f[1], f[end]), freqs, cumpowers)

play_bt = Button(fig[2,3], label="play")
on(play_bt.clicks) do _
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
