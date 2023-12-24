struct QCCenter{A, B, T<:Number}
	Hleft::B
	Hright::B
	adagTleft::Vector{A}
	adagTright::Vector{A}
	Tdagaleft::Vector{A}
	Tdagaright::Vector{A}
	PAleft::Matrix{A}
	PAright::Matrix{A}
	BQleft::Matrix{A}
	BQright::Matrix{A}
	workspace::Vector{T}
end

function QCCenter(left::QCSiteStorages, right::QCSiteStorages)
	# left = renormalizedstorage(left0)
	# right = renormalizedstorage(right0)
	adagTleft, adagTright = filter_pair_vector(left.adagT, right.adagT)
	Tdagaleft, Tdagaright = filter_pair_vector(left.Tdaga, right.Tdaga)
	Hleft, Hright = left.H, right.H
	n = max_blocksize(left.H)
	n = max(max_blocksize(right.H), n)
	workspace = zeros(scalartype(Hleft), n*n)
	return QCCenter(left.H, right.H, adagTleft, adagTright, Tdagaleft, Tdagaright, left.PA, right.PA, left.BQ, right.BQ, workspace)
end

# leftstorage(m::QCCenter) = QCSiteStorages(m.Hleft, m.BQleft, m.PAleft, m.adagTleft, m.Tdagaleft)
# rightstorage(m::QCCenter) = QCSiteStorages(m.Hright, m.BQright, m.PAright, m.adagTright, m.Tdagaright)
max_blocksize(x::AbstractTensorMap) = mapreduce(a->size(a[2], 1), max, blocks(x); init = 0)

function calc_galerkin(m::QCCenter, x::MPSBondTensor)
	out = m(x)
	try
		return norm(leftnull(x)' * out)
	catch
		return norm(out * rightnull(x)' )
	end
end

function TK.mul!(y, m::QCCenter, x) 
	workspace = m.workspace
	mul!(y, m.Hleft, x, true, false)
	mul!(y, x, m.Hright, true, true)
	# @tensor y[1,3] += x[1,2] * m.Hright[2,3]
	for (left, right) in zip(m.adagTleft, m.adagTright)
		apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
	end
	for (left, right) in zip(m.Tdagaleft, m.Tdagaright)
		apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
	end
	@assert size(m.PAleft) == size(m.PAright)
	for r in 1:size(m.PAleft, 1)
		for s in (r+1):size(m.PAleft, 2)
			if isassigned(m.PAleft, r, s) && isassigned(m.PAright, r, s)
				left, right = m.PAleft[r, s], m.PAright[r, s]
				apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
			end
		end
	end
	@assert size(m.BQleft) == size(m.BQright)
	for r in 1:size(m.BQleft, 1)
		if isassigned(m.BQleft, r, r) && isassigned(m.BQright, r, r)
			left, right = m.BQleft[r, r], m.BQright[r, r]
			apply_twosides!(y, x, left, right, workspace, -1, add_adjoint=false)
		end
		for s in (r+1):size(m.BQleft, 2)
			if isassigned(m.BQleft, r, s) && isassigned(m.BQright, r, s)
				left, right = m.BQleft[r, s], m.BQright[r, s]
				apply_twosides!(y, x, left, right, workspace, -1, add_adjoint=true)
			end
		end
	end
	return y	
end

(m::QCCenter)(x) = mul!(similar(x), m, x)

# function (m::QCCenter)(x::MPSBondTensor)
# 	workspace = m.workspace
# 	y = m.Hleft * x
# 	mul!(y, x, m.Hright, true, true)
# 	# @tensor y[1,3] += x[1,2] * m.Hright[2,3]
# 	for (left, right) in zip(m.adagTleft, m.adagTright)
# 		apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
# 	end
# 	for (left, right) in zip(m.Tdagaleft, m.Tdagaright)
# 		apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
# 	end
# 	@assert size(m.PAleft) == size(m.PAright)
# 	for r in 1:size(m.PAleft, 1)
# 		for s in (r+1):size(m.PAleft, 2)
# 			if isassigned(m.PAleft, r, s) && isassigned(m.PAright, r, s)
# 				left, right = m.PAleft[r, s], m.PAright[r, s]
# 				apply_twosides!(y, x, left, right, workspace, add_adjoint=true)
# 			end
# 		end
# 	end
# 	@assert size(m.BQleft) == size(m.BQright)
# 	for r in 1:size(m.BQleft, 1)
# 		if isassigned(m.BQleft, r, r) && isassigned(m.BQright, r, r)
# 			left, right = m.BQleft[r, r], m.BQright[r, r]
# 			apply_twosides!(y, x, left, right, workspace, -1, add_adjoint=false)
# 		end
# 		for s in (r+1):size(m.BQleft, 2)
# 			if isassigned(m.BQleft, r, s) && isassigned(m.BQright, r, s)
# 				left, right = m.BQleft[r, s], m.BQright[r, s]
# 				apply_twosides!(y, x, left, right, workspace, -1, add_adjoint=true)
# 			end
# 		end
# 	end
# 	return y
# end

# function TK.dot(y::MPSBondTensor, m::QCCenter, x::MPSBondTensor)
# 	@tensor energy = m.Hleft[1,2] * x[2,3] * conj(x[1,3])
# 	@tensor energy += x[1,2] * m.Hright[2,3] * conj(x[1,3])
# 	for (left, right) in zip(m.adagTleft, m.adagTright)
# 		@tensor tmp = left[1,2,3] * x[3,4] * right[4,2,5] * conj(x[1,5])
# 		energy += tmp
# 		energy += tmp'
# 	end
# 	for (left, right) in zip(m.Tdagaleft, m.Tdagaright)
# 		@tensor tmp = left[1,2,3] * x[3,4] * right[4,2,5] * conj(x[1,5])
# 		energy += tmp
# 		energy += tmp'		
# 	end
# 	for r in 1:size(m.PAleft, 1)
# 		for s in (r+1):size(m.PAleft, 2)
# 			if isassigned(m.PAleft, r, s) && isassigned(m.PAright, r, s)
# 				left, right = m.PAleft[r, s], m.PAright[r, s]
# 				@tensor tmp = left[1,2,3] * x[3,4] * right[4,2,5] * conj(x[1,5])
# 				energy += tmp
# 				energy += tmp'						
# 			end
# 		end
# 	end	
# 	for r in 1:size(m.BQleft, 1)
# 		if isassigned(m.BQleft, r, r) && isassigned(m.BQright, r, r)
# 			left, right = m.BQleft[r, r], m.BQright[r, r]
# 			@tensor energy -= left[1,2,3] * x[3,4] * right[4,2,5] * conj(x[1,5])
# 		end
# 		for s in (r+1):size(m.BQleft, 2)
# 			if isassigned(m.BQleft, r, s) && isassigned(m.BQright, r, s)
# 				left, right = m.BQleft[r, s], m.BQright[r, s]
# 				@tensor tmp = left[1,2,3] * x[3,4] * right[4,2,5] * conj(x[1,5])
# 				energy -= tmp
# 				energy -= tmp'						
# 			end
# 		end
# 	end
# 	return energy
# end

# to be optimized
function apply_twosides2!(y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
	@tensor y[1,5] += α * left[1,2,3] * x[3,4] * right[4,2,5]
	if add_adjoint
		@tensor y[3,5] += α * conj(left[1,2,3]) * x[1,4] * conj(right[5,2,4])
	end
	return y
end
function apply_twosides!(y::MPSBondTensor, x::MPSBondTensor, left::MPSTensor, right::MPSTensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
	# println(codomain(left), " ", space(x, 1), " ", space(right))
	for (f1l, f1r) in fusiontrees(left)
		ml = StridedView(dropdims(left[f1l, f1r], dims=2))
		f2r = FusionTree((f1l.uncoupled[1],), f1l.uncoupled[1], (false,))
		f2l = FusionTree((f1r.uncoupled[1], conj(f1l.uncoupled[2])), f2r.coupled, (false, false))
		rj = get(right, (f2l, f2r), nothing)
		if !isnothing(rj)
			mr = StridedView(dropdims(rj, dims=2))
			xj = x[f1r, f1r]
			yj = y[f2r, f2r]
			# yj .+= α .* ml * xj * mr
			# s1, s3 = size(ml, 1), size(xj, 2)
			# tmp = sreshape(StridedView(view(workspace, 1:s1*s3)), s1, s3)
			# mul!(tmp, ml, xj, α, false)

			# tmp = mul_workspace!(workspace, ml, xj, α)
			# mul!(yj, tmp, mr, true, true)
			mul_twosides!(yj, ml, xj, mr, α, true, workspace)
			if add_adjoint
				xj = x[f2r, f2r]
				yj = y[f1r, f1r]
				# yj .+= α .* ml' * xj * mr'
				# s1, s3 = size(ml, 2), size(xj, 2)
				# tmp = sreshape(StridedView(view(workspace, 1:s1*s3)), s1, s3)
				# mul!(tmp, ml', xj, α, false)

				# tmp = mul_workspace!(workspace, ml', xj, α)
				# mul!(yj, tmp, mr', true, true)
				mul_twosides!(yj, ml', xj, mr', α, true, workspace)
			end			
		end
	end
	return y
end

# function apply_twosides!(y::RBTensor, x::RBTensor, left::RATensor, right::RATensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
# 	@tensor y[1,2,8,9] += α * left[1,2,3,4,5] * x[4,5,6,7] * right[6,7,3,8,9]
# 	if add_adjoint
# 		@tensor y[4,5,8,9] += α * conj(left[1,2,3,4,5]) * x[1,2,6,7] * conj(right[8,9,3,6,7])
# 	end
# 	return y
# end

# function apply_twosides2!(y::RBTensor, x::RBTensor, left::RATensor, right::RATensor, workspace::Vector, α::Number=1; add_adjoint::Bool)
# 	for (f1l, f1r) in fusiontrees(y)
# 		yj = dropdims(y[f1l, f1r], dims=(2,3))
# 		for (f2l, f2r) in fusiontrees(right)
# 			mr = dropdims(right[f2l,f2r], dims=(1,3,4))
# 			if f2r == f1r
# 				for (f3l, f3r) in fusiontrees(left)
# 					ml = dropdims(left[f3l, f3r], dims=(2,3,5))
# 					if (f3l.uncoupled[1] == f1l.uncoupled[1]) && (f3l.uncoupled[2] == f1l.uncoupled[2]) && (f3l.uncoupled[3] == conj(f2l.uncoupled[3]))
# 						f4r = FusionTree((f2l.uncoupled[1], f2l.uncoupled[2]), f3r.coupled, (true, false))

# 						xja = get(x, (f3r, f4r), nothing)
# 						if !isnothing(xja)
# 							xj = dropdims(xja, dims=(2,3))
# 							tmp = mul_workspace!(workspace, ml, xj, α)
# 							mul!(yj, tmp, mr, true, true)
# 							if add_adjoint
# 								xj2 = dropdims(x[f1l, f1r], dims=(2,3))
# 								yj2 = dropdims(y[f3r, f4r], dims=(2,3))

# 								tmp2 = mul_workspace!(workspace, ml', xj2, α)
# 								mul!(yj2, tmp2, mr', true, true)

# 							end
# 						end

# 					end
# 				end
# 			end
# 		end
# 	end



# 	return y
# end

function filter_pair_vector(a::Vector{A}, b::Vector{A}) where A
	@assert size(a) == size(b)
	anew = A[]
	bnew = A[]
	for i in 1:length(a)
		if isassigned(a, i) && isassigned(b, i)
			push!(anew, a[i])
			push!(bnew, b[i])
		end
	end
	return anew, bnew
end
