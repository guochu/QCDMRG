function _spinless_operators()
	p = GeneralHamiltonians.spin_half_matrices()
	p = Dict(k=>convert(AbelianMatrix, v) for (k, v) in p)
	sp, sm, z = p["+"], p["-"], p["z"]
	I2 = one(z)
	JW = -z
	return sp, sm, I2, JW
end

const adag, a, I2, JW2 = _spinless_operators()
const adagup = kron(adag, I2)
const aup = adagup'
const adagdown = kron(JW2, adag)
const adown = adagdown'
const I4 = kron(I2, I2)
const JW4 = kron(JW2, JW2)

function sqC(sc, ic::Int, iop::Bool)
	@assert (length(sc) == 2) && (sc[2] == sc[1]+1)
	@assert (ic == 1) || (ic == 2)
	if ic == 1
		return iop ? adagup : aup
	else
		return iop ? adagdown : adown
	end
end
function sgnC(sc)
	@assert (length(sc) == 2) && (sc[2] == sc[1]+1)
	return JW4
end 
empty_operator() = AbelianMatrix(Float64, _u1u1_pspace, Dict())
const _empty = empty_operator()

