with pm.Model() as model_det:
    """deterministic model to test the neural network"""
    # reshape x to (1,N)
    xt = x.reshape((1,N))
    # reshape input weights to (20, 1)
    w_in_a1 = network_wts['z_w'].reshape((num_hidden,1))
    b_in_a1 = network_wts['z_b'].reshape((num_hidden,1))
    # z1 = wx+b; sizes (20,1)*(1,N) = (20,N)
    z1 = tt.add(pm.math.dot(w_in_a1, xt), b_in_a1)
    a1 = pm.math.tanh(z1)
    
    # weights for mu have shape (3,20); (3,20) * (20,N) = (3, N)
    w_a1_mu = network_wts['mu_w']
    # bias for mu have shape (3,1)
    b_a1_mu = network_wts['mu_b'].reshape((K,1))
    # mu = w*a1 + b
    mu = tt.add(pm.math.dot(w_a1_mu, a1), b_a1_mu)
    
    # log_sigma analogous to mu
    w_a1_log_sigma = network_wts['log_sigma_w']
    b_a1_log_sigma = network_wts['log_sigma_b'].reshape((K,1))
    log_sigma = tt.add(pm.math.dot(w_a1_log_sigma, a1), b_a1_log_sigma)
    
    # sigma from log_sigma
    sigma = tt.add(pm.math.exp(log_sigma), sigma_shift)
    
    # weight_z analogous to mu and log_sigma
    w_a1_weight_z = network_wts['weight_z_w']
    b_a1_weight_z = network_wts['weight_z_b'].reshape((K,1))
    weight_z = tt.add(pm.math.dot(w_a1_weight_z, a1), b_a1_weight_z)
    
    # weight from weight_z
    weight = softmax(weight_z)
    
    noise = pm.Normal('noise', mu=0.0, sd=1.0, shape=N)
    
    y_obs = pm.NormalMixture('y_obs', w=weight.T, mu=mu.T, sd=sigma.T, observed=y) + noise