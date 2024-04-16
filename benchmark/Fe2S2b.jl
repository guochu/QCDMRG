push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/InfiniteDMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/GeneralHamiltonians/src")
# push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/QCMPO/src")
# using QCMPO

include("../src/includes.jl")

# push!(LOAD_PATH, "../src")
# using QCDMRG

using JSON
using SphericalTensors
using GeneralHamiltonians



function read_data(pathname)
	data = JSON.parsefile(pathname)
	E0 = data["E0"]
	L = data["L"]
	t = data["t"]
	t = [t...]
	v = data["v"]
	v = [v...]
	return E0, reshape(t, (L, L)), reshape(v, (L, L, L, L))
end

read_data() = read_data("Fe2S2.json")

const H10_FCI_ENERGY = -4.7128481828029525

function do_dmrg(env)
	_energies1 = Float64[]
	times = Float64[]
	for n in 1:8
		trunc = truncdimcutoff(D=250, ϵ=1.0e-10, add_back=0)
		alg = QCDMRG2(verbosity=3, trunc=trunc, toleig=1.0e-6, maxitereig=30)
		t = @elapsed push!(_energies1, sweep!(env, alg)[1])
		println("sweep $n takes $t seconds")
		push!(times, t)
	end
	for n in 9:16
		trunc = truncdimcutoff(D=500, ϵ=1.0e-10, add_back=0)
		alg = QCDMRG2(verbosity=3, trunc=trunc, toleig=1.0e-6, maxitereig=30)
		t = @elapsed push!(_energies1, sweep!(env, alg)[1])
		println("sweep $n takes $t seconds")
		push!(times, t)
	end
	for n in 17:32
		trunc = truncdimcutoff(D=1000, ϵ=1.0e-10, add_back=0)
		alg = QCDMRG2(verbosity=3, trunc=trunc, toleig=1.0e-6, maxitereig=30)
		t = @elapsed push!(_energies1, sweep!(env, alg)[1])
		println("sweep $n takes $t seconds")
		push!(times, t)
	end
	return _energies1, times
end


function main()
	E0, t, v = read_data()
	println("E0 = ", E0)
	ham = MolecularHamiltonian(0.5 * (t + t'), 0.5 * v)
	L = length(ham)
	mps = randomqcmps(L; D=50, right=Rep[U₁×U₁]((15, 15)=>1)) # random state
	trunc = truncdimcutoff(D=50, ϵ=1.0e-10, add_back=0)

	# physectors = [(0,0) for i in 1:L]
	# for i in 1:15
	# 	physectors[i] = (1,1)
	# end
	# # general mpo for the Hamiltonian
	# mps0 = prodqcmps(physectors) # HF initial state
	# mps = mps0

	canonicalize!(mps, alg=Orthogonalize(trunc=trunc, normalize=true))
	env = environments(ham, mps)

	eigvalues, times = do_dmrg(env)

	filename = "result/F2eS2b.json"

	eigvalues .+= E0

	results = Dict("times"=>times, "energies"=>eigvalues)

	open(filename, "w") do f
		write(f, JSON.json(results))
	end
end


