using Statistics
using LinearAlgebra

function least_squares(x_hist::Array{Float64, 1}, y_hist::Array{Float64, 1}, lamb_w::Float64 = 1.0)
    @assert length(x_hist) == length(y_hist)
    if all(x_hist .== 1)
        w_pred = sum(y_hist) / (length(x_hist) + lamb_w)
    else
        w_pred = sum(x_hist .* y_hist) / (sum(x_hist .^ 2) + lamb_w)
    end
    return w_pred
end

function train_als(
    X_data::Array{Float64, 2},
    Y_data::Array{Float64, 1},
    Y_pred::Array{Float64, 1},
    b_init::Float64,
    w_init::Array{Float64, 1},
    V_init::Array{Float64, 2},
    max_iter::Int,
    show_progress::Bool,
    hyper_param::Dict
)
    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    N = length(w)
    K = size(V, 2)

    # precompute error and quadratic
    error = Y_pred - Y_data # (D,)
    quad = X_data * V       # (D, K)
    error_hist = Array{Float64, 1}(undef, max_iter + 1)
    error_hist[1] = mean(error .^ 2)

    for iter in 1:max_iter
        # update b
        x_hist = ones(length(X_data[:, 1]))
        y_hist = b .- error
        b_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
        error = error .+ b_new .- b
        b = b_new

        # update w
        for i in 1:N
            x_hist = X_data[:, i]
            y_hist = x_hist .* (w[i] .* x_hist .- error)
            w_i_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
            error = error .+ (w_i_new - w[i]) .* x_hist
            w[i] = w_i_new
        end

        # update V
        for i in 1:N
            for k in 1:K
                G_ik = X_data[:, i] .* (quad[:, k] - V[i, k] .* X_data[:, i])
                x_hist = G_ik
                y_hist = (V[i, k] .* G_ik .- error)
                V_ik_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
                error = error .+ (V_ik_new - V[i, k]) .* G_ik
                quad[:, k] = quad[:, k] .+ (V_ik_new - V[i, k]) .* X_data[:, i]
                V[i, k] = V_ik_new
            end
        end

        error_hist[iter + 1] = mean(error .^ 2)
        if show_progress & ((iter + 1) % 50 == 0)
            println("iter: $(iter + 1), error: $(mean(error .^ 2))")
        end
    end
    return b, w, V, error_hist
end
