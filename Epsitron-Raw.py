class EpsitronTransformer:
    def __init__(self, distribution_bias,  matrix_num, epsitron_bias):
    	self.distribution_bias = distribution_bias
    	self.dot_multi = matrix_num
    	self.entropy_coef = 0.075
    	self.bias = epsitron_bias
    	
    def epsitron_lite_multi_matrix_linear_softmax(self, x):
    	dot_multi = self.dot_multi
    	bias = self.bias
    	
    	uniform = np.ones_like(x) / len(x)
    	kl_divergence = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(uniform))
    	kl_divergence = 0.005 + np.log1p(kl_divergence)
    	curvature = np.mean(np.abs(np.diff(np.diff(x))))
    	sigmoid = 1.0 / (1 - curvature)
    	planner_Q = x + (1.0 / (1 + np.exp(-curvature)))
    	planner_K = x + (planner_Q / kl_divergence)
    	planner_V = x + (planner_K + planner_Q / kl_divergence)
    	
    	all_planner = planner_Q + planner_K + planner_V
    	kl_planner = np.sum(all_planner * np.log(np.clip(all_planner, 1e-8, None)) - np.log(x))
    	kl_planner = sigmoid + np.log1p(kl_planner)
    	
    	kl_divergence = 0.005 + np.log1p(kl_divergence)
    	Q_Curve = np.mean(np.abs(np.diff(np.diff(planner_Q))))
    	K_Curve = np.mean(np.abs(np.diff(np.diff(planner_K))))
    	V_Curve = np.mean(np.abs(np.diff(np.diff(planner_V))))
    	all_curve = sigmoid + Q_Curve + K_Curve + V_Curve
    	
    	efficient_kl = kl_planner / kl_divergence 
    	kl_curve = efficient_kl / all_curve	
    	concluded = (sigmoid + dot_multi) - bias / efficient_kl

    	Q = np.dot(planner_Q, concluded)
    	K = np.dot(planner_K, concluded)
    	V = np.dot(planner_V, concluded)

    	Q = np.nan_to_num(Q, nan=0.0, posinf=1e40, neginf=1e-40)    	
    	K = np.nan_to_num(K, nan=0.0, posinf=1e40, neginf=1e-40)    
    	V = np.nan_to_num(V, nan=0.0, posinf=1e40, neginf=1e-40)
    	
    	return Q, K, V  
    	

    	
    	 	 	
    def epsitron_lite_linear_attention(self, x):
    	Q, K, V= self.epsitron_lite_multi_matrix_linear_softmax(x)          	
    	uniform = np.ones_like(x) / len(x)
    	kl_divergence = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(uniform))
    	kl_divergence = 0.005 + np.log1p(kl_divergence)
    	curvature = np.mean(np.abs(np.diff(np.diff(x))))
    	sigmoid = 1.0 / (1 - curvature)
    	
    	kl_Q = np.sum(Q * np.log(np.clip(Q, 1e-8, None)) - np.log(x))
    	kl_Q = sigmoid + np.log1p(kl_Q)
    	kl_K = np.sum(K * np.log(np.clip(K, 1e-8, None)) - np.log(x))
    	kl_K = sigmoid + np.log1p(kl_K)  
    	kl_V = np.sum(V * np.log(np.clip(V, 1e-8, None)) - np.log(x))
    	kl_V = sigmoid + np.log1p(kl_V)    
    	
    	Q = sigmoid + np.log1p(Q)    	
    	K = sigmoid + np.log1p(K)
    	V = sigmoid + np.log1p(V)
    	mat_Q_product = np.dot(Q, kl_Q)
    	mat_K_product = np.dot(K, kl_K)   
    	mat_V_product = np.dot(V, kl_V) 
    		
    	mat_divergence = mat_Q_product + mat_K_product + mat_V_product / kl_divergence
    	mat_compare = mat_divergence / kl_divergence      	
    	weight_value_product_divergence = mat_compare / kl_V
    	mat_sum_divergence = mat_divergence / weight_value_product_divergence   	
    	efficient_kl = sigmoid + (kl_Q + kl_K + kl_V) / kl_divergence
    	weight_value_product_divergence /= efficient_kl

    	entropy_regularization = np.sum(-Q * np.log(np.clip(-Q, 1e-8, None)) - np.log(V))    	
    	entropy_regularization = sigmoid + np.log1p(entropy_regularization)
    	entropy_loss = entropy_regularization / efficient_kl
    	mat_entropy = mat_sum_divergence / entropy_loss    
    	mat_curvature = mat_sum_divergence / curvature 
    		    	
    	x += weight_value_product_divergence / mat_curvature
    	x /= mat_entropy 
    	x += sigmoid        	
    	if np.isnan(x).any() or not np.isfinite(x).any():
    		x = np.ones_like(x)
    		
    	return x
    	
    	 				  	    		  	    			  	    		  	    			
 
    
          
    def epsitronic_chipper(self, x, entropy=None):
    	
    	distribution_bias = self.distribution_bias
    	raw_curve = 0.005 + np.mean(np.abs(np.diff(np.diff(x))))
    	uniform = np.ones_like(x) / len(x)
    	kl_divergence = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(uniform))
    	kl_divergence = 0.005 + np.log1p(kl_divergence)
    	sigmoid = 1.0 / (1 - raw_curve)
    	raw_scaling = np.sum(x) / np.mean(x)
    	weight = np.exp(np.log1p(x))
    	weight /= raw_scaling 
    	weight_curve = sigmoid + np.mean(np.abs(np.diff(np.diff(weight))))
    	weighted = np.sum(weight)   	    	
    	uniform = np.ones_like(weight) / len(weight)
    	
    	x += weighted + (distribution_bias - entropy) / weight_curve
    	x /= distribution_bias 
    	x += sigmoid 
 	        	
    	if np.isnan(x).any() or not np.isfinite(x).any():
    	    	x = np.ones_like(x) / len(x)
 	
    	return x
    	  	    	    	
    	    	    	
    def layer_norm(self, x, eps=1e-5):
    	mean = np.mean(x, axis=-1, keepdims=True)
    	std = np.std(x, axis=-1, keepdims=True)
    	return (x - mean) / (std + eps)
        	        	
    def epsitron_softmax(self, x, distribution_coef):
    	entropy_coef = self.entropy_coef
    	distribution_bias = self.distribution_bias
    	raw_noise = np.std(x)
    	var_noise = np.var(x)	
    	curvature = 0.005 + np.mean(np.abs(np.diff(np.diff(x))))
    	uniform = np.ones_like(x) / len(x)
    	kl_divergence = np.sum(x * np.log(np.clip(x, 1e-8, None) - np.log(uniform)))
    	kl_divergence = 0.005 + np.log1p(kl_divergence)
    	sigmoid = 1.0 / (1 - curvature)
    	delta = raw_noise / sigmoid
    	delta /= var_noise + sigmoid
    	
    	meta_first = np.exp(np.log1p(x))
    	sec_meta = np.exp(np.log1p(meta_first))
    	third_meta = np.exp(np.log1p(sec_meta))
    	
    	first_curve = np.mean(np.abs(np.diff(np.diff(meta_first)))) 
    	sec_curve = np.mean(np.abs(np.diff(np.diff(sec_meta))))   	    	    	
    	third_curve = np.mean(np.abs(np.diff(np.diff(third_meta)))) 
    	raw_weights = np.sum(x) / np.mean(x)
    	gradient_weights = np.sum(first_curve + sec_meta + third_meta) / kl_divergence
    	gradient_descent = kl_divergence / sigmoid + first_curve + sec_curve + third_curve
    	efficient_descent = gradient_weights / gradient_descent  
    	efficient_distribution = (gradient_weights + raw_weights) / efficient_descent - distribution_bias  
    	  	 
    	x = x + (1 - entropy_coef) / np.log1p(gradient_weights)
    	x *= (efficient_descent + efficient_distribution) / distribution_coef + sigmoid  	 	
    	x = np.clip(x, 1e-8, None)	    	
    	x = self.epsitron_recalibrator_softmax(x)    
    		
    	if np.isnan(x).any() or not np.isfinite(x).any():
    	   	x = (np.ones_like(x) / len(x))  
    	   	   
    
    	return x
    	
    	
    def epsitron_multi_badge_softmax(self, logit, constant=0.005):
    	calibrator = self.epsitron_recalibrator_softmax(logit)
    	uniform = np.ones_like(logit) / len(logit)
    	
    	kl_divergence = np.sum(logit * np.log(np.clip(logit, 1e-8,None)) - np.log(uniform))
    	kl_divergence = constant + np.log1p(kl_divergence) 	
    	raw_curve = constant + np.mean(np.abs(np.diff(np.diff(logit))))
    	sigmoid = 1.0 / (1 - raw_curve)

    	first_meta = np.exp(np.log1p(logit))
    	sec_meta = np.exp(np.log1p(first_meta))    	
    	third_meta = np.exp(np.log1p(sec_meta))
    	all_meta = first_meta + sec_meta + third_meta
    	weight_divergence = np.sum(all_meta) / kl_divergence
    	kl_meta_divergence = np.sum(all_meta * np.log(np.clip(all_meta, 1e-8, None)) - np.log(uniform))
    	kl_meta_divergence = constant + np.log1p(kl_meta_divergence)
    		
    	first_curve = np.mean(np.abs(np.diff(np.diff(first_meta))))
    	sec_curve = np.mean(np.abs(np.diff(np.diff(sec_meta))))   
    	third_curve = np.mean(np.abs(np.diff(np.diff(third_meta))))    
    	all_curve = sigmoid + first_curve + sec_curve + third_curve
    	
    	efficient_kl = kl_meta_divergence / kl_divergence 
    	kl_descent_manifold = efficient_kl / all_curve
    	weight_curved_manifold = weight_divergence / all_curve	
    	first_weight_descent = np.sum(first_meta) / kl_descent_manifold 
    	sec_weight_descent = np.sum(sec_meta) / kl_descent_manifold
    	third_weight_descent = np.sum(third_meta) / kl_descent_manifold 
    	
    	new_logit = logit + calibrator
    	new_logit /= weight_curved_manifold 
    	new_logit /= kl_descent_manifold
    	new_logit += sigmoid
    	 	
    	Q = np.dot(logit, first_weight_descent)
    	K = np.dot(all_meta, sec_weight_descent)
    	V = np.dot(new_logit, third_weight_descent)
    	Q /= kl_descent_manifold + 1e-8
    	K /= kl_descent_manifold + 1e-8
    	V /= kl_descent_manifold + 1e-8
    	
    	Q = np.nan_to_num(Q, nan=0.0, posinf=1e40, neginf=1e-40)    	
    	K = np.nan_to_num(K, nan=0.0, posinf=1e40, neginf=1e-40)    
    	V = np.nan_to_num(V, nan=0.0, posinf=1e40, neginf=1e-40)
        	
    	return Q, K, V
    	
    	
    def epsitron_stable_attention(self, x):
    	constant = 0.005
    	Q, K, V = self.epsitron_multi_badge_softmax(x, constant=0.005)
    	uniform = np.ones_like(x) / len(x)
    	Q_kl = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(Q)) 
    	K_kl = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(K)) 
    	V_kl = np.sum(x * np.log(np.clip(x, 1e-8, None)) - np.log(V)) 
    	Q_kl = constant + np.log1p(Q_kl)   
    	K_kl = constant + np.log1p(K_kl) 
    	V_kl = constant + np.log1p(V_kl)
    	all_query_div = constant + (Q_kl + K_kl / V_kl)
    	curvature = np.mean(np.abs(np.diff(np.diff(x))))    	
    	sigmoid = 1.0 / (1 - curvature)   
    			
    	Q_planner = np.exp(np.log1p(Q))
    	K_planner = np.exp(np.log1p(K))
    	V_planner = np.exp(np.log1p(V))
    	Q_divergence = np.sum(Q_planner * np.log(np.clip(Q_planner, 1e-8, None)) - np.log(Q))
    	Q_divergence = sigmoid + np.log1p(Q_divergence)
    	K_divergence = np.sum(K_planner * np.log(np.clip(K_planner, 1e-8, None)) - np.log(K))
    	K_divergence = sigmoid + np.log1p(K_divergence)    	
    	V_divergence = np.sum(V_planner * np.log(np.clip(V_planner, 1e-8, None)) - np.log(V))
    	V_divergence = sigmoid + np.log1p(V_divergence)    	
    	concluded_divergence = sigmoid + (Q_divergence + K_divergence / V_divergence)
    	
    	Q_curve = np.mean(np.abs(np.diff(np.diff(Q_planner))))
    	K_curve = np.mean(np.abs(np.diff(np.diff(K_planner))))
    	V_curve = np.mean(np.abs(np.diff(np.diff(V_planner))))
    	all_curve = sigmoid  + Q_curve + K_curve + V_curve  
    	
    	efficient_kl = concluded_divergence / all_query_div
    	kl_curve = efficient_kl / all_curve
    	weight_divergence = concluded_divergence / all_curve
    	
    	x += weight_divergence / kl_curve
    	x /= efficient_kl
    	x += sigmoid 
    	
    	if np.isnan(x).any() or not np.isfinite(x).any():
    		x = np.ones_like(x) / len(x)
    		
    	return x
    	