using Statistics
using Random
using Distributions

function least_squares(x_hist::Array{Float64}, y_hist::Array{Float64}, var_param::Float64 = 1.0)
    @assert length(x_hist) == length(y_hist)
    if all(x_hist .== 1)
        w_pred = sum(y_hist) / (length(x_hist) + 1/var_param)
    else
        w_pred = sum(x_hist .* y_hist) / (sum(x_hist .^ 2) + 1/var_param)
    end
    w_pred
end

function sample_noise_var(x_hist::Array{Float64}, y_hist::Array{Float64}, model_param::Float64, hyper_param::Dict)
    @assert length(x_hist) == length(y_hist)
    alpha_noise = hyper_param["alpha_noise"]
    beta_noise = hyper_param["beta_noise"]
    D = length(x_hist)
    alpha_noise_post = alpha_noise + D / 2
    beta_noise_post = beta_noise + sum((y_hist - x_hist * model_param).^2) / 2
    1 / rand(Gamma(alpha_noise_post, 1/beta_noise_post))
end

function sample_parameter(x_hist::Array{Float64}, y_hist::Array{Float64}, var_noise::Float64, mean_param::Float64, var_param::Float64)
    var_param_post = 1 / (sum(x_hist .^ 2) / var_noise + 1/var_param)
    mean_param_post = var_param_post * (sum(x_hist .* y_hist) / var_noise + mean_param/var_param)
    rand(Normal(mean_param_post, sqrt(var_param_post)))
end

function sample_mean_parameter(model_params::Array{Float64}, var_param::Float64, hyper_param::Dict)
    mean_mean_param = hyper_param["mean_mean_param"]
    precision_mean_param = hyper_param["precision_mean_param"]
    N = length(model_params)
    precision_mean_param_post = precision_mean_param + N
    mean_mean_param_post = (sum(model_params) + precision_mean_param * mean_mean_param) / precision_mean_param_post
    rand(Normal(mean_mean_param_post, sqrt(var_param/precision_mean_param_post)))
end

function sample_var_parameter(model_params::Array{Float64}, mean_param::Float64, hyper_param::Dict)
    alpha_param = hyper_param["alpha_param"]
    beta_param = hyper_param["beta_param"]
    N = length(model_params)
    alpha_param_post = alpha_param + (N + 1) / 2
    beta_param_post = beta_param + (sum((model_params .- mean_param).^2) + hyper_param["precision_mean_param"] * (mean_param - hyper_param["mean_mean_param"])^2) / 2
    1 / rand(Gamma(alpha_param_post, 1/beta_param_post))
end

function train_bayes(X_data::Array{Float64, 2}, Y_data::Array{Float64}, Y_pred::Array{Float64}, b_init::Float64, w_init::Array{Float64}, V_init::Array{Float64, 2}, als_iter::Int, max_iter::Int, hyper_param::Dict)
    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    D = size(X_data, 1)
    N = size(X_data, 2)
    K = size(V, 2)

    var_w = 1.0
    mean_w = mean(w)
    var_v = ones(K)
    mean_v = mean(V, dims=1)

    error = Y_data - Y_pred
    quad = X_data * V
    error_hist = Array{Float64}(undef, max_iter + 1)
    error_hist[1] = mean(error .^ 2)

    for iter in 1:max_iter
        x_hist = ones(D)
        y_hist = b .- error
        if iter < als_iter
            b_new = least_squares(x_hist, y_hist, hyper_param["var_b"])
        else
            var_noise = sample_noise_var(x_hist, y_hist, b, hyper_param)
            b_new = sample_parameter(x_hist, y_hist, var_noise, hyper_param["mean_b"], hyper_param["var_b"])
        end
        error = error .+ b_new .- b
        b = b_new

        if iter >= als_iter
            var_w = sample_var_parameter(w, mean_w, hyper_param)
            mean_w = sample_mean_parameter(w, var_w, hyper_param)
        end
        for i in 1:N
            x_hist = X_data[:, i]
            y_hist = x_hist .* (w[i] .* x_hist .- error)
            if iter < als_iter
                w_i_new = least_squares(x_hist, y_hist, var_w)
            else
                var_noise = sample_noise_var(x_hist, y_hist, w[i], hyper_param)
                w_i_new = sample_parameter(x_hist, y_hist, var_noise, mean_w, var_w)
            end
            error = error .+ (w_i_new - w[i]) .* x_hist
            w[i] = w_i_new
        end

        for k in 1:K
            if iter >= als_iter
                var_v[k] = sample_var_parameter(V[:, k], mean_v[k], hyper_param)
                mean_v[k] = sample_mean_parameter(V[:, k], var_v[k], hyper_param)
            end
            for i in 1:N
                G_ik = X_data[:, i] .* (quad[:, k] - V[i, k] .* X_data[:, i])
                x_hist = G_ik
                y_hist = (V[i, k] .* G_ik .- error)
                if iter < als_iter
                    V_ik_new = least_squares(x_hist, y_hist, var_v[k])
                else
                    var_noise = sample_noise_var(x_hist, y_hist, V[i, k], hyper_param)
                    V_ik_new = sample_parameter(x_hist, y_hist, var_noise, mean_v[k], var_v[k])
                end
                error = error .+ (V_ik_new - V[i, k]) .* G_ik
                quad[:, k] = quad[:, k] .+ (V_ik_new - V[i, k]) .* X_data[:, i]
            end
        end
        error_hist[iter + 1] = mean(error .^ 2)
        if (iter + 1) % 50 == 0
            println("iter: $(iter+1), error: $(mean(error .^ 2))")
        end
    end
    return b, w, V, error_hist
end
