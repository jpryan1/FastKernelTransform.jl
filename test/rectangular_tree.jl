module TestTree
using LinearAlgebra
using FastKernelTransform
using FastKernelTransform: BallNode, Tree, initialize_tree
using Test

@testset "rectangular tree" begin
    d, n, m = 3, 256, 512
    x, y = randn(d, n), randn(d, m)
    x, y = [c for c in eachcol(x)], [c for c in eachcol(y)]
    max_dofs = 16
    T = initialize_tree(x, y, max_dofs, 3)
    for (i, node) in enumerate(T.allnodes)
        @test i == node.node_index
    end
end

end
