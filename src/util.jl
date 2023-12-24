u1u1_pspace() = Rep[U₁×U₁]((0,0)=>1, (0,1)=>1, (1,0)=>1, (1,1)=>1)
const _u1u1_pspace = u1u1_pspace()

fermionparity(s::U1Irrep) = isodd(convert(Int, s.charge))
fermionparity(s::SU2Irrep) = false
fermionparity(P::ProductSector) = mapreduce(fermionparity, xor, P.sectors)



function totensormap(m::AbelianMatrix{S}; side::Symbol=:L) where {S <: GradedSpace} 
	((side == :L) || (side == :R)) || throw(ArgumentError("side must be :L or :R"))
	phy = physical_space(m)
	if side == :R
		left_sectors = sectortype(S)[]
		for (k, v) in m.data
			ko, ki = k
			left = first(ki ⊗ conj(ko))
			push!(left_sectors, left)
		end
		vspace = S(item=>1 for item in unique(left_sectors))
		r = TensorMap(ds->zeros(scalartype(m), ds), phy ⊗ vspace ← phy )
		for (k, v) in m.data
			ko, ki = k
			for _l in sectors(vspace)
				if first(ko ⊗ _l) == ki
					# println(ko, " ", _l, " ", ki)
					copyto!(r[(ko, _l, conj(ki))], v)
				end
			end
		end
	else
		r = flip2(totensormap(m, side=:R))
	end
	return r	
end

function tomatrixmap(m::AbelianMatrix{S}) where {S <: GradedSpace} 
	phy = physical_space(m)
	r = TensorMap(ds->zeros(scalartype(m), ds), phy ← phy )
	for (k, v) in m.data
		ko, ki = k
		if ko == ki
			blocks(r)[ko] = v
		else
			iszero(v) || error(ArgumentError("input operator should be diagonal"))
		end
	end
	return r	
end

function flip2(t::MPSTensor)
	vspace = space(t, 2)
	F = isomorphism(storagetype(t), flip(vspace), vspace)
	@tensor t2[3,1;4] := F[1,2] * t[3,2,4]
end

# function phy_dagger(t::MPSTensor)
# 	t′ = t'
# 	return flip2(permute(t′, (1,3), (2,)))
# end
function phy_dagger(t::MPSTensor)
	t′ = t'
	vspace = space(t′, 3)
	F = isomorphism(storagetype(t), flip(vspace), vspace)
	@tensor r[3,1;4] := F[1,2] * t′[3,4,2]
end

function _issymmetric(t::AbstractTensorMap{S, N, N}; atol::Real=1.0e-10) where {S, N}
	(domain(t) == codomain(t)) || throw(SpaceMismatch())
	r = t - t'
	return norm(r) < atol
end


randomqcmps(::Type{T}, L::Int; kwargs...) where {T<:Number} = randommps(T, [_u1u1_pspace for i in 1:L]; kwargs...)
randomqcmps(L::Int; kwargs...) = randomqcmps(Float64, L; kwargs...)

