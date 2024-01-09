# push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/SphericalTensors/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/InfiniteDMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/GeneralHamiltonians/src")

include("../src/includes.jl")

using JSON
using SphericalTensors



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

function do_dmrg(env, alg)
	_energies1 = Float64[]
	times = Float64[]
	for n in 1:5
		t = @elapsed push!(_energies1, sweep!(env, alg)[1])
		println("sweep $n takes $t seconds")
		push!(times, t)
	end
	return _energies1, times
end


function main(D)
	E0, t, v = read_data()
	println("E0 = ", E0)
	ham = MolecularHamiltonian(0.5 * (t + t'), 0.5 * v)
	L = length(ham)
	# mps = randomqcmps(L; D=D, right=Rep[U₁×U₁]((15, 15)=>1)) # random state
	physectors = [(0,0) for i in 1:L]
	for i in 1:15
		physectors[i] = (1,1)
	end
	mps = prodqcmps(physectors) # HF initial state
	canonicalize!(mps)
	env = environments(ham, mps)

	alg = DMRG2(verbosity=3, trunc=truncdimcutoff(D=D, ϵ=1.0e-10))
	eigvalues, times = do_dmrg(env, alg)

	filename = "result/F2eS2_D$(D).json"

	eigvalues .+= E0

	results = Dict("times"=>times, "energies"=>eigvalues)

	open(filename, "w") do f
		write(f, JSON.json(results))
	end
end


