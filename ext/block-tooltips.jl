# https://github.com/MakieOrg/Makie.jl/pull/4451

"""
    function tooltip!(b::Block, str::AbstractString; enabled=true, delay=0, depth=9e3, kwargs...)
Adds a tooltip to a block.  `delay` specifies the interval in seconds before the
tooltip appears if `enabled` is `true`.  `depth` should be large to ensure that
the tooltip is in front.  See `tooltip` for more details.
# Examples
```julia-repl
julia> f = Figure()
julia> t = Toggle(f[1,1])
Toggle()
julia> tooltip!(t, "I'm a Toggle")
Plot{Makie.tooltip, Tuple{Vec{2, Float32}, String}}
julia> b = Button(f[2,1])
Button()
julia> e = Observable(false)
Observable(false)
julia> tt = tooltip!(b, "I'm a Button", placement = :below, enabled = e, delay = 1)
Plot{Makie.tooltip, Tuple{Vec{2, Float32}, String}}
julia> e[] = true
:always
```
"""
function tooltip!(b::Makie.Block, str::AbstractString; enabled=true, delay=0, depth=9e3, kwargs...)
    _enabled = typeof(enabled)<:Observable ? enabled : Observable(enabled)
    _delay = typeof(delay)<:Observable ? delay : Observable(delay)
    _depth = typeof(depth)<:Observable ? depth : Observable(depth)

    position = Observable(Point2f(0))
    tt = Makie.tooltip!(b.blockscene, position, str; visible=false, kwargs...)
    on(z->translate!(tt, 0, 0, z), _depth)

    function update_viz0(mp, bbox)
        if mp in bbox
            position[] = mp
            tt.visible[] = true
        else
            tt.visible[] = false
        end
    end

    if _delay[] > 0
        t0, last_mp = time(), b.blockscene.events.mouseposition[]
    end

    function update_viz(mp, bbox)
        if mp in bbox
            last_mp in bbox || (t0 = time())
            last_mp = mp
            position[] = mp
            tt.visible[] = time() > t0 + _delay[]
        else
            last_mp = mp
            tt.visible[] = false
        end
    end

    was_open = false
    channel = Channel{Tuple}(Inf) do ch
        for (mp,bbox) in ch
            if isopen(b.blockscene)
                was_open = true
                _delay[]==0 ? update_viz0(mp,bbox) : update_viz(mp,bbox)
            end
            !isopen(b.blockscene) && was_open && break
        end
    end

    obsfun = nothing
    on(_enabled) do e
        if e && isnothing(obsfun)
            obsfun = onany(b.blockscene.events.mouseposition, b.layoutobservables.computedbbox) do mp, bbox
                Makie.empty_channel!(channel)
                put!(channel, (mp,bbox))
            end
        elseif !e && !isnothing(obsfun)
            foreach(off, obsfun)
            obsfun = nothing
            tt.visible[] = false
        end
    end

    notify(_enabled)
    notify(_delay)
    notify(_depth)
    notify(b.blockscene.events.mouseposition)
    return tt
end
