using Random
using LinearAlgebra
using Statistics
using Distributions

function sample_noise_var(x_hist, y_hist, model_param, hyper_param)
    alpha_noise = hyper_param["alpha_noise"]
    beta_noise = hyper_param["beta_noise"]
    D = length(x_hist)
    alpha_noise_post = alpha_noise + D / 2
    beta_noise_post = beta_noise + sum((y_hist - x_hist .* model_param).^2) / 2
    var_noise = 1 / rand(Gamma(alpha_noise_post, 1 / beta_noise_post))
    return var_noise
end

function sample_parameter(x_hist, y_hist, var_noise, mean_param, var_param)
    var_param_post = 1 / ( sum(x_hist .^ 2) / var_noise + 1 / var_param )
    mean_param_post = var_param_post * (sum(x_hist .* y_hist) / var_noise + mean_param / var_param)
    return rand(Normal(mean_param_post, sqrt(var_param_post)))
end

function sample_mean_parameter(model_params, var_param, hyper_param)
    mean_mean_param = hyper_param["mean_mean_param"]
    precision_mean_param = hyper_param["precision_mean_param"]
    N = length(model_params)
    precision_mean_param_post = precision_mean_param + N
    mean_mean_param_post = (sum(model_params) + precision_mean_param * mean_mean_param) / precision_mean_param_post
    return rand(Normal(mean_mean_param_post, sqrt(var_param / precision_mean_param_post)))
end

function sample_var_parameter(model_params, mean_param, hyper_param)
    mean_mean_param = hyper_param["mean_mean_param"]
    precision_mean_param = hyper_param["precision_mean_param"]
    alpha_param = hyper_param["alpha_param"]
    beta_param = hyper_param["beta_param"]
    N = length(model_params)
    alpha_param_post = alpha_param + (N + 1) / 2
    beta_param_post = beta_param + (sum((model_params .- mean_param).^2) + precision_mean_param * (mean_param - mean_mean_param)^2) / 2
    return 1 / rand(Gamma(alpha_param_post, 1 / beta_param_post))
end

function train_bayes(X_data, Y_data, Y_pred, b_init, w_init, V_init, max_iter, show_progress, hyper_param, seed = nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    D, N = size(X_data)
    K = size(V, 2)

    # place holder
    var_w = 1.0
    mean_w = 0.0
    var_v = ones(K)
    mean_v = zeros(K)

    # precompute error and quadratic
    error = Y_pred .- Y_data
    quad = X_data * V
    error_hist = zeros(max_iter + 1)
    error_hist[1] = mean(error .^ 2)

    for iter in 1:max_iter
        # update hyperparameters
        var_noise = sample_noise_var(ones(D), b .- error, b, hyper_param)

        var_w = sample_var_parameter(w, mean_w, hyper_param)
        mean_w = sample_mean_parameter(w, var_w, hyper_param)
        for k in 1:K
            var_v[k] = sample_var_parameter(V[:, k], mean_v[k], hyper_param)
            mean_v[k] = sample_mean_parameter(V[:, k], var_v[k], hyper_param)
        end

        # update parameters
        x_hist = ones(D)
        y_hist = b .- error
        b_new = sample_parameter(x_hist, y_hist, var_noise, hyper_param["mean_b"], hyper_param["var_b"])
        error .+= b_new - b
        b = b_new

        for i in 1:N
            x_hist = X_data[:, i]
            y_hist = X_data[:, i] .* (w[i] .* X_data[:, i] .- error)
            w_i_new = sample_parameter(x_hist, y_hist, var_noise, mean_w, var_w)
            error .+= (w_i_new - w[i]) .* X_data[:, i]
            w[i] = w_i_new
        end

        for k in 1:K
            for i in 1:N
                G_ik = X_data[:, i] .* (quad[:, k] .- V[i, k] .* X_data[:, i])
                x_hist = G_ik
                y_hist = (V[i, k] .* G_ik .- error)
                V_ik_new = sample_parameter(x_hist, y_hist, var_noise, mean_v[k], var_v[k])
                error .+= (V_ik_new - V[i, k]) .* G_ik
                quad[:, k] .+= (V_ik_new - V[i, k]) .* X_data[:, i]
                V[i, k] = V_ik_new
            end
        end

        error_hist[iter + 1] = mean(error .^ 2)
        if show_progress && (iter + 1) % 50 == 0
            println("iter: $(iter + 1), error: $(mean(error .^ 2))")
        end
    end

    return b, w, V, error_hist
end
