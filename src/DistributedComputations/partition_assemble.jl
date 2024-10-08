import Oceananigans.Architectures: on_architecture

all_reduce(op, val, arch::Distributed) = 
    MPI.Allreduce(val, op, arch.communicator)

all_reduce(op, val, arch) = val

# MPI Barrier
barrier!(arch) = nothing
barrier!(arch::Distributed) = MPI.Barrier(arch.communicator)

"""
    concatenate_local_sizes(n, arch::Distributed) 

Return a 3-Tuple containing a vector of `size(grid, idx)` for each rank in 
all 3 directions.
"""
concatenate_local_sizes(n, arch::Distributed) = 
    Tuple(concatenate_local_sizes(n, arch, i) for i in 1:length(n))

function concatenate_local_sizes(n, arch::Distributed, idx)
    R = arch.ranks[idx]
    r = arch.local_index[idx]
    n = n isa Number ? n : n[idx]
    l = zeros(Int, R)

    r1, r2 = arch.local_index[[1, 2, 3] .!= idx]
    
    if r1 == 1 && r2 == 1
        l[r] = n
    end

    MPI.Allreduce!(l, +, arch.communicator)
    
    return l
end

# Partitioning (localization of global objects) and assembly (global assembly of local objects)
# Used for grid constructors (cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z)
# We need to repeat the value at the right boundary
function partition_coordinate(c::AbstractVector, n, arch, idx)
    nl = concatenate_local_sizes(n, arch, idx)
    r  = arch.local_index[idx]

    start_idx = sum(nl[1:r-1]) + 1 # sum of all previous rank's dimension + 1
    end_idx   = if r == ranks(arch)[idx]
        length(c)
    else
        sum(nl[1:r]) + 1 
    end

    return c[start_idx : end_idx]
end

function partition_coordinate(c::Tuple, n, arch, idx)
    nl = concatenate_local_sizes(n, arch, idx)
    N  = sum(nl)
    R  = arch.ranks[idx]
    Δl = (c[2] - c[1]) / N  

    l = Tuple{Float64, Float64}[(c[1], c[1] + Δl * nl[1])]
    for i in 2:R
        lp = l[i-1][2]
        push!(l, (lp, lp + Δl * nl[i]))
    end

    return l[arch.local_index[idx]]
end

"""
    assemble_coordinate(c::AbstractVector, n, R, r, r1, r2, comm) 

Builds a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture. Since we use a global reduction, only ranks at positions
1 in the other two directions `r1 == 1` and `r2 == 1` fill the 1D array.
"""
function assemble_coordinate(c_local::AbstractVector, n, arch, idx) 
    nl = concatenate_local_sizes(n, arch, idx)
    R  = arch.ranks[idx]
    r  = arch.local_index[idx]
    r2 = [arch.local_index[i] for i in filter(x -> x != idx, (1, 2, 3))]

    c_global = zeros(eltype(c_local), sum(nl)+1)

    if r2[1] == 1 && r2[2] == 1
        c_global[1 + sum(nl[1:r-1]) : sum(nl[1:r])] .= c_local[1:end-1]
        r == R && (c_global[end] = c_local[end])
    end

    MPI.Allreduce!(c_global, +, arch.communicator)

    return c_global
end

# Simple case, just take the first and the last core
function assemble_coordinate(c_local::Tuple, n, arch, idx) 
    c_global = zeros(Float64, 2)
    
    rank = arch.local_index
    R    = arch.ranks[idx]
    r    = rank[idx]
    r2   = [rank[i] for i in filter(x -> x != idx, (1, 2, 3))]

    if rank[1] == 1 && rank[2] == 1 && rank[3] == 1
        c_global[1] = c_local[1]
    elseif r == R && r2[1] == 1 && r2[1] == 1
        c_global[2] = c_local[2]
    end

    MPI.Allreduce!(c_global, +, arch.communicator)

    return tuple(c_global...)
end 

# TODO: partition_global_array and construct_global_array
# do not currently work for 3D parallelizations
# (They are not used anywhere in the code at the moment exept for immersed boundaries)
"""
    partition_global_array(arch, c_global, (nx, ny, nz))

Partition a global array in local arrays of size `(nx, ny)` if 2D or `(nx, ny, nz)` is 3D.
Usefull for boundary arrays, forcings and initial conditions.
"""
partition_global_array(arch, c_global::AbstractArray, n) = c_global
partition_global_array(arch, c_global::Function, n)      = c_global 

# Here we assume that we cannot partition in z (we should remove support for that)
function partition_global_array(arch::Distributed, c_global::AbstractArray, n) 
    c_global = on_architecture(CPU(), c_global)

    ri, rj, rk = arch.local_index

    dims = length(size(c_global))
    nx, ny, nz = concatenate_local_sizes(n, arch)

    nz = nz[1]

    if dims == 2 
        c_local = zeros(eltype(c_global), nx[ri], ny[rj])

        c_local .= c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                            1 + sum(ny[1:rj-1]) : sum(ny[1:rj])]
    else
        c_local = zeros(eltype(c_global), nx[ri], ny[rj], nz)

        c_local .= c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                            1 + sum(ny[1:rj-1]) : sum(ny[1:rj]), 
                            1:nz]
    end
    return on_architecture(child_architecture(arch), c_local)
end

"""
    construct_global_array(arch, c_local, (nx, ny, nz))

Construct global array from local arrays (2D of size `(nx, ny)` or 3D of size (`nx, ny, nz`)).
Usefull for boundary arrays, forcings and initial conditions.
"""
construct_global_array(arch, c_local::AbstractArray, n) = c_local
construct_global_array(arch, c_local::Function, N)      = c_local

# TODO: This does not work for 3D parallelizations!!!
function construct_global_array(arch::Distributed, c_local::AbstractArray, n) 
    c_local = on_architecture(CPU(), c_local)

    ri, rj, rk = arch.local_index

    dims = length(size(c_local))

    nx, ny, nz = concatenate_local_sizes(n, arch)

    Nx = sum(nx)
    Ny = sum(ny)
    Nz = nz[1]

    if dims == 2 
        c_global = zeros(eltype(c_local), Nx, Ny)
    
        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj])] .= c_local[1:nx[ri], 1:ny[rj]]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    else
        c_global = zeros(eltype(c_local), Nx, Ny, Nz)

        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj]),
                 1:Nz] .= c_local[1:nx[ri], 1:ny[rj], 1:Nz]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    end

    return on_architecture(child_architecture(arch), c_global)
end
