function DMRG.leftsweep!(env::QCDMRGCache, alg::DMRG2)
	energies = Float64[]
	delta = 0.
	for bond in 1:length(env)-2
		(alg.verbosity > 3) && println("sweeping from left to right at bond: $bond")
		mpsA, mpsB = env.mps[bond], env.mps[bond+1]
		# two-site effective Hamiltonian
		Sleft = renormalizestorageleft(env, bond, space_l(mpsA))
		Sright = renormalizestorageright(env, bond+1, space_r(mpsB)')
		Opleft = renormalizedstorage(Sleft)
		Opright  = renormalizedstorage(Sright)
		heff = QCCenter(Opleft, Opright)
		# initial guess
		@tensor x[1,2; 4,5] := mpsA[1,2,3] * mpsB[3,4,5]
		# eigenvalue_0, eigenvec_0 = _eigsolve(heff, renormalizedoperator(x), alg.maxitereig, alg.toleig)
		eigenvalues_0, eigenvecs_0 = eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(;  maxiter=100, tol=alg.toleig, eager=true))
		eigenvalue_0, eigenvec_0 = eigenvalues_0[1], eigenvecs_0[1]
		eigenvec = TensorMap(blocks(eigenvec_0), codomain(x), domain(x))
		u, s, v, err = tsvd!(eigenvec, trunc=alg.trunc)
		# compute error
		normalize!(s)
		v2 = s * v
		x′ = u * v2
		err_1 = dot(x′, x)
		delta = max(delta,abs(1-abs(err_1)))
		# recalcute the energy
		x2′ = renormalizedoperator(x′)
		eigenvalue = dot(x2′, heff(x2′))
		(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
		
		# update storages
		env.mps[bond] = u
		env.mps[bond+1] = permute(v2, (1,2), (3,))
		Snew = updatestoragerenormalizeleft(Opleft, renormalizedoperator(env.mps[bond]))
		setstorage!(env, bond, Snew)
		push!(energies, eigenvalue)
	end
	return energies, delta
end

function DMRG.rightsweep!(env::QCDMRGCache, alg::DMRG2)
	energies = Float64[]
	delta = 0.
	for bond in length(env)-1:-1:1
		(alg.verbosity > 3) && println("sweeping from right to left at bond: $bond")
		mpsA, mpsB = env.mps[bond], env.mps[bond+1]
		# two-site effective Hamiltonian
		Sleft = renormalizestorageleft(env, bond, space_l(mpsA))
		Sright = renormalizestorageright(env, bond+1, space_r(mpsB)')
		Opleft = renormalizedstorage(Sleft)
		Opright  = renormalizedstorage(Sright)
		heff = QCCenter(Opleft, Opright)
		# initial guess
		@tensor x[1,2; 4,5] := mpsA[1,2,3] * mpsB[3,4,5]
		# eigenvalue_0, eigenvec_0 = _eigsolve(heff, renormalizedoperator(x), alg.maxitereig, alg.toleig)
		eigenvalues_0, eigenvecs_0 = eigsolve(heff, renormalizedoperator(x), 1, :SR, Lanczos(; maxiter=100, tol=alg.toleig, eager=true))
		eigenvalue_0, eigenvec_0 = eigenvalues_0[1], eigenvecs_0[1]
		eigenvec = TensorMap(blocks(eigenvec_0), codomain(x), domain(x))
		u, s, v, err = tsvd!(eigenvec, trunc=alg.trunc)
		# compute error
		normalize!(s)
		u2 = u * s 
		x′ = u2 * v
		err_1 = dot(x′, x)
		delta = max(delta,abs(1-abs(err_1)))
		x2′ = renormalizedoperator(x′)
		eigenvalue = dot(x2′, heff(x2′))
		(alg.verbosity > 2) && println("E₀=$(eigenvalue_0), E=$eigenvalue, δ=$(round(delta, digits=12)), χ=$(dim(space(s, 2))) after optimizing bond $bond")
		# update storages
		env.mps[bond] = u2
		mpsB = v 
		env.mps[bond+1] = permute(mpsB, (1,2), (3,))
		env.mps.s[bond+1] = s
		Snew = updatestoragerenormalizeright(Opright, renormalizedoperator(mpsB))
		setstorage!(env, bond+1, Snew)
		push!(energies, eigenvalue)
	end
	return energies, delta	
end

function DMRG.sweep!(m::QCDMRGCache, alg::DMRG2)
	Energies1, delta1 = leftsweep!(m, alg)
	Energies2, delta2 = rightsweep!(m, alg)
	delta = max(delta1, delta2)
	if alg.verbosity > 1
		println("E=$(Energies2[end]), δ=$(round(delta, digits=12)) after a full sweep")
		println()
	end
	return Energies2[end], delta
end

function _eigsolve(h, init, maxiter, tol)
	if dim(init) >= 20
		eigenvalue_0, eigenvec_0, info = DMRG.simple_lanczos_solver(h, init, "SR", maxiter, tol, verbosity=0)
	else
		init = TensorMap(randn, scalartype(init), space(init))
		eigenvalues, eigenvecs, infos = eigsolve(h, init, 1, :SR, Lanczos())
		# (infos.converged >= 1) 
		eigenvalue_0 = eigenvalues[1]
		eigenvec_0 = eigenvecs[1]
	end
	return eigenvalue_0, eigenvec_0
end
