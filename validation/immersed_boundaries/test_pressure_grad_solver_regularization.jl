
using Printf
using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, 
                    fft_poisson_solver, FourierTridiagonalPoissonSolver, AsymptoticPoissonPreconditioner
using Oceananigans.Utils: prettytime
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Oceananigans.Operators: divᶜᶜᶜ

## Simulation parameters
Nx = 32
Ny = 32

tᶠ = 5days # simulation run time
Δtᵒ = 30#180minutes # interval for saving output
H = 600meters # vertical extent
L = 2*2H # horizontal extent

# uniform grid works without blowing up
# z_faces_array = [0., 200., 400., 600.]
# nonuniform grid blows up
# z_faces_array = [range(0,300,13); range(330,600,6)]
# z_faces_array = [0., 20., 50., 90., 140., 200., 270., 350., 440., 540., 600.]
z_faces_array = [0., 200., 270., 350., 440., 540., 600.]

## Create vertical grid
# Creates a vertical grid with near-constant spacing `refinement * Lz / Nz` near the bottom:
# "Warped" coordinate
Nz = 15
kwarp(k, N) = (N + 1 - k) / N
# Linear near-surface generator
ζ(k, N, refinement) = 1 + (kwarp(k, N) - 1) / refinement
# Bottom-intensified stretching function
Σ(k, N, stretching) = (1 - exp(-stretching * kwarp(k, N))) / (1 - exp(-stretching))
# Generating function
z_faces(k) = - H * (ζ(k, Nz, 1.2) * Σ(k, Nz, 15) - 1)


grid = RectilinearGrid(size=(Nx, Ny, length(z_faces_array)-1), 
        x = (0, L),
        y = (0, L), 
        # z = z_faces_array,
        z = z_faces,
        halo = (4,4,4),
        topology = (Periodic, Periodic, Bounded)
)
h = 200meters # topographic height
topography = zeros(Nx,Ny)
topography[Nx÷2,Ny÷2] = h
grid_immerse = ImmersedBoundaryGrid(grid, GridFittedBottom(topography))

# IC such that flow is in phase with predicted linear response, but otherwise quiescent
U₀=0.01
uᵢ(x, y, z) = -U₀
vᵢ(x, y, z) = 0.
# preconditioner = fft_poisson_solver(grid)
preconditioner = AsymptoticPoissonPreconditioner()
regularizer = 1
tol = 1e-9
model = NonhydrostaticModel(
    grid = grid_immerse,
    # comment out the pressure_solver when using deafult FFT solver
    pressure_solver = ConjugateGradientPoissonSolver(grid_immerse; maxiter=100,preconditioner,
                                                    reltol=tol, abstol=tol)
)
set!(model, u=uᵢ, v=vᵢ)

Δt = 30
simulation = Simulation(model, Δt = Δt, stop_time = tᶠ)

u, v, w = model.velocities

udiv = KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, model.grid, u, v, w)
fname = string("test_CGsolver_with_immersed_nonuniformgrid_CG_", tol, "anisotropy.nc")
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (udiv=udiv,u=u,v=v,w=w),
                                        schedule = TimeInterval(Δtᵒ),
                                        filename = fname,
                                        overwrite_existing = true)
## Progress messages
progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|u|: %.2e, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.u),maximum(abs, model.velocities.w)) 
simulation.callbacks[:progress] = Callback(progress_message, TimeInterval(Δt))
run!(simulation)



# ## plotting animation to see where blows up

# using CairoMakie
# using NCDatasets
# using Printf
# fname = "test_CGsolver_with_immersed_nonuniformgrid_CG.nc"

# ds = Dataset(fname,"r")

# # grids
# zC = ds["zC"]; Nz=length(zC)
# zF = ds["zF"]; #Nz=length(zF)
# xC = ds["xC"]; Nx=length(xC)
# xF = ds["xF"];

# yC = ds["yC"]; Ny=length(yC)
# t = ds["time"];

# u = ds["u"][:,:,:,:];
# w = ds["w"][:,:,:,:];
# udiv = ds["udiv"][:,:,:,:];

# u_center = (u[:,:,:,:].+vcat(u[2:end,:,:,:], u[1:1,:,:,:]))./2
# w_center = (w[:,:,1:end-1,:].+w[:,:,2:end,:])./2
# u_center[u_center.==0].=NaN
# w_center[w_center.==0].=NaN
# w[w.==0].=NaN
# u[u.==0].=NaN


# # plot
# n = Observable(1)
# uₙ = @lift(u_center[:,16,:,$n])
# wₙ = @lift(w_center[:,16,:,$n])
# udivₙ = @lift(udiv[:,16,:,$n])

# fig = Figure(resolution = (1000, 1000), figure_padding=(10, 40, 10, 10), size=(600,800),fontsize=20)
# axis_kwargs = (xlabel = "x (m)",
#                   ylabel = "z (m)",
#                   limits = ((0, ds["xF"][end]), (0, ds["zF"][end])),
#                   )
# title = @lift @sprintf("t=%1.2f hrs", t[$n]/3600)
# fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)
                  
                  
# ax_u = Axis(fig[2, 1]; title = "u", axis_kwargs...)
# ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
# ax_udiv = Axis(fig[4, 1]; title = L"∇⋅\vec{u}", axis_kwargs...)



# using ColorSchemes
# U₀ = 0.01
# hm_u = heatmap!(ax_u, xC[:], zC[:], uₙ,
#     colorrange = (-U₀, U₀), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
#     nan_color = :gray)
# hm_w = heatmap!(ax_w, xC[:], zC[:], wₙ,
#     colorrange = (-U₀, U₀), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
#     nan_color = :gray)
# Colorbar(fig[3,2], hm_w; label = "m/s")

# hm_udiv = heatmap!(ax_udiv, xC[:], zC[:], udivₙ,
#     colorrange = (-1e-8,1e-8), colormap = :diverging_bwr_20_95_c54_n256,
#     lowclip=cgrad(:diverging_bwr_20_95_c54_n256)[1], highclip=cgrad(:diverging_bwr_20_95_c54_n256)[end],
#     nan_color = :gray)
# Colorbar(fig[4,2], hm_udiv; label = "1/s")


# frames =  (1:50:length(t))

# filename = join(split(fname, ".")[1:end-1], ".")

# record(fig, string(filename,".mp4"), frames, framerate=23) do i
#     @info "Plotting frame $i of $(frames[end])..."
#     n[] = i
# end
