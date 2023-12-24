function _to_spin_orbitals(h1e::AbstractMatrix{<:Real})
	k = size(h1e, 1)
	h1e′ = zeros(Float64, (2*k, 2*k))
	for i in 1:k, j in 1:k
		h1e′[2*i-1, 2*j-1] = h1e[i, j]
		h1e′[2*i, 2*j] = h1e[i, j]
	end
	return h1e′
end

function _to_spin_orbitals(h2e::AbstractArray{<:Real, 4})
	k = size(h2e, 1)
	h2e′ = zeros(Float64, (2*k, 2*k, 2*k, 2*k))
	for p in 1:k, q in 1:k, r in 1:k, s in 1:k
		h2e′[2*p-1, 2*q-1, 2*r-1, 2*s-1] = h2e[p,q,r,s]
		h2e′[2*p, 2*q, 2*r, 2*s] = h2e[p,q,r,s]
		h2e′[2*p-1, 2*q, 2*r, 2*s-1] = h2e[p,q,r,s]
		h2e′[2*p, 2*q-1, 2*r-1, 2*s] = h2e[p,q,r,s]
	end
	return h2e′
end

get_spin_orbitals(h1e::AbstractMatrix{<:Real}, h2e::AbstractArray{<:Real, 4}) = _to_spin_orbitals(h1e), _to_spin_orbitals(h2e)


# antisymmetrize operation does not commute with to_spin_orbitals
# one must first do to_spin_orbitals and then do antisymmetrize!!!
function antisymmetrize(eri::AbstractArray{<:Real, 4})
	L = size(eri, 1)
	h2e = 0.5 .* (eri + permutedims(eri, (3,4,1,2)))
	d = zeros(size(h2e))
	for i in 1:L
		d[i, i, :, :] = h2e[i, i, :, :]
	end
	h2e = 0.5 * (h2e - permutedims(h2e, (2,1,3,4)))
	h2e .+= d
	d = fill!(d, 0)
	for i in 1:L
		d[:, :, i, i] = h2e[:, :, i, i]
	end
	h2e = 0.5 * (h2e - permutedims(h2e, (1,2,4,3)))
	h2e .+= d
	return h2e
end
"""
	remove_antisymmetric(eri::AbstractArray{<:Real, 4})

Return an equivalent two-body integral, with restriction
p < q and r < s (0 if not satisfied)
"""
function remove_antisymmetric(eri::AbstractArray{<:Real, 4})
	k = size(eri, 1)
	h2e = zeros(size(eri))
	for p in 1:k
		for q in (p+1):k
			for r in 1:k
				for s in (r+1):k
					h2e[p, q, r, s] = 4 * eri[p, q, r, s]
				end
			end
		end
	end
	for p in 1:k
		for r in 1:k
			for s in (r+1):k
				h2e[p, p, r, s] = 2 * eri[p, p, r, s]
			end
		end
	end
	for p in 1:k
		for q in (p+1):k
			for r in 1:k
				h2e[p, q, r, r] = 2 * eri[p, q, r, r]
			end
		end
	end
	for p in 1:k
		for r in 1:k
			h2e[p, p, r, r] = eri[r, r, r, r]
		end
	end
	return h2e
end