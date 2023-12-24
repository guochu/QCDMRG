println("------------------------------------")
println("|        Renormalize operator      |")
println("------------------------------------")


@testset "test RenormalizedATensor" begin

	vspace = Rep[U₁ × U₁]((0,0)=>1, (0, 1)=>2, (1,0)=>2, (1,1)=>3, (1,2)=>4, (2,1)=>5)
	pspace = Rep[U₁ × U₁]((0,0)=>1, (0, 1)=>1, (1,0)=>1, (1,1)=>2)

	# t1 = RenormalizedATensor(zeros, vspace ⊗ pspace ⊗ pspace, vspace ⊗ pspace)
	t2 = TensorMap(randn, vspace ⊗ pspace ⊗ pspace, vspace ⊗ pspace)
	
	@test convert(TensorMap, convert(RenormalizedATensor, t2)) ≈ t2
end
