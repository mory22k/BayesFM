using Statistics

function train_als(
    X_data::Array{Float64,2},
    Y_data::Array{Float64,1},
    Y_pred::Array{Float64,1},
    b::Float64,
    w::Array{Float64,1},
    V::Array{Float64,2},
    max_iter::Int=100,
    lamb_b::Float64=1.0,
    lamb_w::Float64=1.0,
    lamb_v::Float64=1.0
    )::Tuple{Float64, Array{Float64,1}, Array{Float64,2}, Array{Float64,1}}

    N = size(w, 1)
    K = size(V, 2)

    error_hist = Array{Float64,1}(undef, max_iter + 1)
    error = Y_data - Y_pred
    quad = X_data * V
    error_hist[1] = mean(error .^ 2)

    for iter in 1:max_iter
        b_new = sum(b .- error) / (lamb_b + size(X_data, 2))
        error .+= b_new - b
        b = b_new

        for i in 1:N
            w_i_new = sum(X_data[:,i] .* (w[i] * X_data[:,i] - error)) / (lamb_w + sum(X_data[:,i] .^ 2))
            error .+= (w_i_new - w[i]) .* X_data[:,i]
            w[i] = w_i_new
        end

        for i in 1:N
            for k in 1:K
                G_ik = X_data[:,i] .* (quad[:,k] - V[i,k] * X_data[:,i])
                V_ik_new = sum(G_ik .* (V[i,k] * G_ik - error)) / (lamb_v + sum(G_ik .^ 2))
                error .+= (V_ik_new - V[i,k]) .* G_ik
                quad[:,k] .= quad[:,k] + (V_ik_new - V[i,k]) .* X_data[:,i]
            end
        end

        error_hist[iter+1] = mean(error .^ 2)
        println("iter: ", iter, ", error: ", mean(error .^ 2))
    end
    return b, w, V, error_hist
end
