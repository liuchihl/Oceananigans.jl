
# Create a snapshot of arrows at a specific x-position
snapshot_n = 1  # You can change this to any index you want

fig_snapshot = CairoMakie.Figure(resolution = (800, 600))
ax_snapshot = Axis(fig_snapshot[1, 1];
    xlabel = "y (m)",
    ylabel = "z (m)",
    title = "Velocity vectors at x = $(round(xC[snapshot_n]/1e3, digits=1)) km",
    limits = ((0, yF[end]), (0, zF[end]*10)))

arrows!(ax_snapshot,
    yC[1:15:end], 10*zC[1:15:end],
    v_cen[snapshot_n,1:15:end,1:15:end],
    100*w[snapshot_n,1:15:end,1:15:end],
    arrowsize = 10,
    lengthscale = 5.0,  # Increased lengthscale to make tails more visible
    linewidth = 1.2,
    color = :black)

save(string("output/", simname, "/velocity_vectors_snapshot_",simname,"_tᶠ=",tᶠ,".png"), fig_snapshot)
