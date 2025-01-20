from cymetric.models.fubinistudy import FSModel
import gc
import tensorflow as tf
import tensorflow.keras as tfk
from laplacian_funcs import *
from custom_networks import batch_process_helper_func
from BetaModel import *
import os
import numpy as np

class HarmonicFormModel(FSModel):
    r"""FreeModel from which all other models inherit.

    The training and validation steps are implemented in this class. All
    other computational routines are inherited from:
    cymetric.models.fubinistudy.FSModel
    
    Example:
        Assume that `BASIS` and `data` have been generated with a point 
        generator.

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from cymetric.models.tfmodels import FreeModel
        >>> from cymetric.models.tfhelper import prepare_tf_basis
        >>> tfk = tf.keras
        >>> data = np.load('dataset.npz')
        >>> BASIS = prepare_tf_basis(np.load('basis.pickle', allow_pickle=True))
    
        set up the nn and FreeModel

        >>> nfold = 3
        >>> ncoords = data['X_train'].shape[1]
        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(nfold**2),
        ...     ]
        ... )
        >>> model = FreeModel(nn, BASIS)

        next we can compile and train

        >>> from cymetric.models.metrics import TotalLoss
        >>> metrics = [TotalLoss()]
        >>> opt = tfk.optimizers.Adam()
        >>> model.compile(custom_metrics = metrics, optimizer = opt)
        >>> model.fit(data['X_train'], data['y_train'], epochs=1)

        For other custom metrics and callbacks to be tracked, check
        :py:mod:`cymetric.models.metrics` and
        :py:mod:`cymetric.models.callbacks`.
    """
    def __init__(self, tfmodel, BASIS,HYMmetric,linebundleforHYM,functionforbaseharmonicform_jbar,alpha=None, **kwargs):
        r"""FreeModel is a tensorflow model predicting CY metrics. 
        
        The output is
            
            .. math:: g_{\text{out}} = g_{\text{NN}}
        
        a hermitian (nfold, nfold) tensor with each float directly predicted
        from the neural network.

        NOTE:
            * The model by default does not train against the ricci loss.
                
                To enable ricci training, set `self.learn_ricci = True`,
                **before** the tracing process. For validation data 
                `self.learn_ricci_val = True`,
                can be modified separately.

            * The models loss contributions are

                1. sigma_loss
                2. kaehler loss
                3. transition loss
                4. ricci loss (disabled)
                5. volk loss

            * The different losses are weighted with alpha.

            * The (FB-) norms for each loss are specified with the keyword-arg

                >>> model = FreeModel(nn, BASIS, norm = [1. for _ in range(5)])

            * Set kappa to the kappa value of your training data.

                >>> kappa = np.mean(data['y_train'][:,-2])

        Args:
            tfmodel (tfk.model): the underlying neural network.
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from cymetric.pointgen.pointgen.
            alpha ([5//NLOSS], float): Weighting of each loss contribution.
                Defaults to None, which corresponds to equal weights.
        """
        super().__init__(BASIS=BASIS, **kwargs)
        self.model = tfmodel
        self.NLOSS = 2
        # variable or constant or just tensor?
        if alpha is not None:
            #self.alpha = [tf.Variable(a, dtype=tf.float32) for a in alpha]
            self.alpha = [a for a in alpha]
        else:
            #self.alpha = [tf.Variable(1., dtype=tf.float32) for _ in range(self.NLOSS)]
            self.alpha = [1. for _ in range(self.NLOSS)]
        self.learn_transition = tf.cast(True, dtype=tf.bool)
        self.learn_laplacian = tf.cast(True, dtype=tf.bool)
        #self.functionforbaseharmonicform_bbar=functionforbaseharmonicform_bbar #the H_FS which we are correcting
        #self.source_function_for_coclosure=source_function_for_coclosure#for type-2 vs type-1 this will differ
        self.HYMmetric = HYMmetric
        self.functionforbaseharmonicform_jbar = tf.function(functionforbaseharmonicform_jbar)
        #self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        #self.learn_volk = tf.cast(False, dtype=tf.bool)

        self.custom_metrics = None
        #self.kappa = tf.cast(BASIS['KAPPA'], dtype=tf.float32)
        self.gclipping = float(5.0)
        self.linebundleforHYM=linebundleforHYM
        # add to compile?
        #self.sigma_loss = sigma_loss(self.kappa, tf.cast(self.nfold, dtype=tf.float32))

    def weighting_function_for_section(self,coordsfortrans):
        #$\frac{(\tanh{20\log(2x)}+1)(\tanh{(-20\log(x/2))}+1)}{4}$
        return (tf.math.tanh(20*tf.math.log(2*coordsfortrans))+1)*(tf.math.tanh(-20*tf.math.log(coordsfortrans/2))+1)/4
    
    def get_section_transition_to_patch_mask(self, points, patch_mask):
        r"""Given the old points points, and patch_mask giving the new patch, compute the transition functions for s_i - t_ij s_j.
        we wish to have s_i = psi_ij s_j. Consider f(a1,a2,a3) on P2. Define the local sections f1(1,x1,x2), f2(y1,1,y2) and f3(z1,z2,1)
        Then f1 = y1^-k f2 = x1^k f2, f1 = z1**(-k)f3 = x2**(k) f3
        """
        coordsfortrans = tf.boolean_mask(points, patch_mask) # picks out only the values which we  wish to set to 1 under the patch_mask
        coordsfortrans = tf.reshape(coordsfortrans, (-1, self.nProjective)) # reshape to the appropriate form
        # which is (bSize, nProjective) in this case, so (bSize, 4)
        coordsfortranstotheki=tf.math.reduce_prod(coordsfortrans**self.linebundleforHYM,-1)
        considerforweightfunction=tf.cast(tf.cast(self.linebundleforHYM,tf.bool),tf.float32)
        weights=self.weighting_function_for_section(tf.math.abs(coordsfortrans)**considerforweightfunction)
        #raise to the 0 or 1th power, depending on whether the line bundle is zero in that direction
        weights = tf.reduce_prod(weights, -1)# so this should basically yield 1 if the relevant coordinates are in the [0.2,2] belt, and 0 otherwise

        return coordsfortranstotheki,weights
    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the harmonic form model, we demand that the section transforms as a section of the line bundle to which it belongs. \phi(\lambda^q_i z_i)=\phi(z_i)
        also can separately check that the 1-form itself transforms appropriately?

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, self.nProjective))
        current_patch_mask = self._indices_to_mask(patch_indices)
        #fix the ones we want to eliminate - no point in patching these
        fixed = self._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        if self.nhyper == 1:
            other_patches = tf.gather(self.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = self._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, self.nProjective))
        other_patch_mask = self._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)#expanded points
        patch_points = self._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool)) # other patches
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        sigmaj = self(real_patch_points, training=True)
        sigmai = tf.repeat(self(points), self.nTransitions, axis=0)
        # this takes (1,z1,1,z2,1,z3,1,z4), picks out the ones we want to set to 1 - i.e. z1,z2,z3,z4 for the (1,1,1,1) patch,
        # and returns z1^k1 x z2^k2 etc... It therefore should multiply the w (i.e. the sigmaj)?
        transformation,weights=self.get_section_transition_to_patch_mask(exp_points,other_patch_mask) 
        all_t_loss = tf.math.abs(sigmai-transformation*sigmaj)*weights
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[1], axis=-1)
        #print("weights meannnn: ")
        #tf.print(tf.reduce_mean(weights)/self.nTransitions)
        #print("weights mean zero: ")
        #tf.print(tf.reduce_mean(weights**0.)/self.nTransitions)
        return all_t_loss/(self.nTransitions)



    def compute_laplacian_loss(self,x,pullbacks,invmetrics,sources,training=False):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        #def directly_compute_source(cpoints,invmetrics,pullbacks):
        #    z0 = cpoints[:,4]
        #    z1 = cpoints[:,5]
        #    w0 = cpoints[:,6]
        #    w1 = cpoints[:,7]
        #    w0bar = tf.math.conj(w0)
        #    w1bar = tf.math.conj(w1)
        #    z0bar = tf.math.conj(z0)
        #    z1bar = tf.math.conj(z1)
        #    kappaz=tf.cast(tf.math.abs(z0)**2+tf.math.abs(z1)**2,tf.complex64)
        #    kappaw=tf.cast(tf.math.abs(w0)**2+tf.math.abs(w1)**2,tf.complex64)
        #    eye = tf.eye(8,dtype=tf.complex64)
        #    dmu_w_over_w0 = (tf.expand_dims(w0,1)*eye[:,7] - tf.expand_dims(w1,1)*eye[:,6])/tf.expand_dims(w0,1)**2
        #    dmu_z_over_z0 = (tf.expand_dims(z0,1)*eye[:,5] - tf.expand_dims(z1,1)*eye[:,4])/tf.expand_dims(z0,1)**2
        #    dmubar_w_over_w0  = tf.math.conj(dmu_w_over_w0)
        #    dmubar_z_over_z0= tf.math.conj(dmu_z_over_z0)
        #    g1 = tf.einsum('xba,xbj,xai,xj,xi->x',invmetrics,tf.math.conj(pullbacks),pullbacks,dmubar_w_over_w0,dmu_w_over_w0)
        #    g2 = tf.einsum('xba,xbj,xai,xj,xi->x',invmetrics,tf.math.conj(pullbacks),pullbacks,dmubar_w_over_w0,dmu_z_over_z0)
        #    poly1 = kappaz**(-1)*kappaw**(-2)*w0**2 * w1bar*g1*z1
        #    poly2 = kappaz**(-2)*kappaw**(-1)*z0bar * w0 * z0**2 * g2
        #
        #    #return kappaz**(-3)*(z1 * w1 + z1**2*w1)
        #    return 3*w0bar**2 *w1bar**2 *(-poly1+poly2)

        #def directly_compute_source_true(cpoints,invmetrics,pullbacks):
        #    z0 = cpoints[:,4]
        #    z1 = cpoints[:,5]
        #    w0 = cpoints[:,6]
        #    w1 = cpoints[:,7]
        #    w0bar = tf.math.conj(w0)
        #    w1bar = tf.math.conj(w1)
        #    z0bar = tf.math.conj(z0)
        #    z1bar = tf.math.conj(z1)
        #    kappaz=tf.cast(tf.math.abs(z0)**2+tf.math.abs(z1)**2,tf.complex64)
        #    kappaw=tf.cast(tf.math.abs(w0)**2+tf.math.abs(w1)**2,tf.complex64)
        #    eye = tf.eye(8,dtype=tf.complex64)
        #    dmu_w_over_w0 = (tf.expand_dims(w0,1)*eye[:,7] - tf.expand_dims(w1,1)*eye[:,6])/tf.expand_dims(w0,1)**2
        #    dmu_z_over_z0 = (tf.expand_dims(z0,1)*eye[:,5] - tf.expand_dims(z1,1)*eye[:,4])/tf.expand_dims(z0,1)**2
        #    dmubar_w_over_w0  = tf.math.conj(dmu_w_over_w0)
        #    dmubar_z_over_z0= tf.math.conj(dmu_z_over_z0)
        #    g1 = tf.einsum('xba,xbj,xai,xj,xi->x',invmetrics,tf.math.conj(pullbacks),pullbacks,dmubar_w_over_w0,dmu_z_over_z0)
        #    poly = w0bar**2 * z0 * (-kappaz**(-2)* z1bar* (z0*w1bar + w0bar*z1) + w0bar/kappaz)*g1
        #
        #    return -1*poly




        ##print("fixed now????")
        ##lpl_losses=tf.math.abs(laplacian(self,x,pullbacks,invmetrics)-(sources))
        lpl_losses=tf.math.abs(laplacianWithH(self,x,pullbacks,invmetrics,self.HYMmetric,training=training)-(sources))
        #complex_pt = point_vec_to_complex(x)
        ##kappa_3 = tf.reduce_sum(complex_pt[:,4:6]*tf.math.conj(complex_pt[:,4:6]),axis=-1)
        ##kappa_4 = tf.reduce_sum(complex_pt[:,6:8]*tf.math.conj(complex_pt[:,6:8]),axis=-1)
        ##print('test polynomial 6')
        ##print('test polynomial 3')
        ##test_polynomial = 100*complex_pt[:,4]*tf.math.conj(complex_pt[:,6])**6 *complex_pt[:,6]**3/kappa_4**6
        ######test_polynomial = (complex_pt[:,4]*tf.math.conj(complex_pt[:,6])**3/kappa_4**3) * kappa_4**(3) * kappa_3**(-1)
        ##lpl_losses=tf.math.abs(self(x)-test_polynomial)
        ##print("laplacian on test poly with additional kappa4 and kappa 3??")
        ##print('test random section learn directly')
        ##z0 = complex_pt[:,4]
        ##z1 = complex_pt[:,5]
        ##w0 = complex_pt[:,6]
        ##w1 = complex_pt[:,7]


        #def sigma_test_fn(pointstensor,**kwargs):
        #    cpoints=point_vec_to_complex(pointstensor)
        #    #return tf.reduce_sum(cpoints**3,axis=-1)
        #    z0 = cpoints[:,4]
        #    z1 = cpoints[:,5]
        #    w0 = cpoints[:,6]
        #    w1 = cpoints[:,7]
        #    w0bar = tf.math.conj(w0)
        #    w1bar = tf.math.conj(w1)
        #    kappaw=tf.cast(tf.math.abs(w0)**2+tf.math.abs(w1)**2,tf.complex64)
        #    return kappaw**(-3)*(z1 * w1bar**3)# + z1*w1bar**3)


        ##sigmatest = kappa_4**(-3) *z1 * tf.math.conj(w1)**3
        ###sigmatest2
        #sources_analytic  = directly_compute_source_true(complex_pt,invmetrics,pullbacks)
        #print("\n", 'real pt', x[0])
        #print('sources analytic')
        #tf.print(sources_analytic[0:5])
        #print(sources_analytic[0:5])
        #print("\n")
        #print('sources numerically computed')
        #tf.print(sources[0:5])
        #print(sources[0:5])

        #difference = tf.abs(test_sources - batched_result_subset)
        #max_diff = tf.reduce_max(difference)
        #print(f"Maximum difference between batched and unbatched computation: {max_diff}")
        ##lpl_losses=tf.math.abs(laplacianWithH(self,x,pullbacks,invmetrics,self.HYMmetric)-(test_polynomial))
        #lpl_losses=tf.math.abs(laplacianWithH(self,x,pullbacks,invmetrics,self.HYMmetric)-sources_analytic)
        #lpl_losses=tf.math.abs(self(x)-(sigmatest))
        #pt = tf.constant([[ 3.2848217e-02,  1.0000000e+00,  3.2474488e-01,  1.0000000e+00,
        # 1.0000000e+00,  4.6791604e-01,  1.0000000e+00,  5.0267313e-02,
        #-8.0492121e-01, -5.1052545e-18,  2.7090704e-01,  6.2252607e-17,
        # 4.1675801e-17, -5.0190330e-02,  2.2366057e-17, -1.5960541e-01]])

        all_lpl_loss = lpl_losses**self.n[0]
        return all_lpl_loss



    def call(self, input_tensor, training=False, j_elim=None):
        r"""Prediction of the NN.

        .. math:: g_{\text{out}} = g_{\text{NN}}

        The additional arguments are included for inheritance reasons.

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                Not used in this model. Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        # nn prediction
        #print("called model")
        #print(input_tensor.dtype)
        #print(type(input_tensor))
        #cpoints=point_vec_to_complex(input_tensor)
        #return self.functionforbaseharmonicform_bbar(self.pg,cpoints)+self.model(input_tensor, training=training)
        #return (lambda x: tf.complex(x[:,0],x[:,1]))(self.model(input_tensor, training=training))# old version
        return self.model(input_tensor, training=training)

    @tf.function
    def corrected_harmonicform(self,input_tensor):
        cpoints=point_vec_to_complex(input_tensor)
        dbarsigma = extder_jbar_for_sigma(input_tensor,self)#note that exter_jbar takes real points, and the thing it differentiates also takes real points
        NuAmbient=self.functionforbaseharmonicform_jbar(cpoints) + dbarsigma
        #self.pullbacks takes real points
        NuCY=tf.einsum('xbj,xj->xb',tf.math.conj(self.pullbacks(input_tensor)),NuAmbient)
        return NuCY
    def uncorrected_FS_harmonicform(self,input_tensor):
        cpoints=point_vec_to_complex(input_tensor)
        NuAmbient=self.functionforbaseharmonicform_jbar(cpoints) 
        #self.pullbacks takes real points
        NuCY=tf.einsum('xbj,xj->xb',tf.math.conj(self.pullbacks(input_tensor)),NuAmbient)
        return NuCY

    def compile(self, custom_metrics=None, **kwargs):
        r"""Compiles the model.
        kwargs takes any argument of regular `tf.model.compile()`
        Example:
            >>> model = FreeModel(nn, BASIS)
            >>> from cymetric.models.metrics import TotalLoss
            >>> metrics = [TotalLoss()]
            >>> opt = tfk.optimizers.Adam()
            >>> model.compile(metrics=metrics, optimizer=opt)
        Args:
            custom_metrics (list, optional): List of custom metrics.
                See also :py:mod:`cymetric.models.metrics`. If None, no metrics
                are tracked during training. Defaults to None.
        """
        if custom_metrics is not None:
            kwargs['metrics'] = custom_metrics
        super(HarmonicFormModel, self).compile(**kwargs)

    @property
    def metrics(self):
        r"""Returns the model's metrics, including custom metrics.
        Returns:
            list: metrics
        """
        return self._metrics

    def train_step(self, data):
        r"""Train step of a single batch in model.fit().

        NOTE:
            1. The first epoch will take additional time, due to tracing.
            
            2. Warnings are plentiful. Disable on your own risk with 

                >>> tf.get_logger().setLevel('ERROR')
            
            3. The conditionals need to be set before tracing. 
            
            4. We employ under the hood gradient clipping.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # if len(data) == 3:
        #     x, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = dataX_train, y_train, train_pullback,inv_mets_train,sources_train
        x = data["X_train"]
        y = None
        y_pred=None
        # print("hi")
        # print(x.shape)
        # print(len(x))
        # The 'y_train/val' arrays contain the integration weights and $\\Omega \\wedge \\bar\\Omega$ for each point. In principle, they can be used for any relevant pointwise information that could be needed during the training process."

        sample_weight = None
        pbs = data["train_pullbacks"]
        invmets = data["inv_mets_train"]
        sources = data["sources_train"]
        #x,sample_weight, pbs, invmets, sources = data#.values()
        # print("help")
        # print(type(data))
        # print(type(data.values()))
        # print(data)
        # print("hi")
        # print(list(x))
        with tf.GradientTape(persistent=False) as tape:
            trainable_vars = self.model.trainable_variables
            #tape.watch(trainable_vars)
            #automatically watch trainable
            # add other loss contributions.
            if self.learn_transition:
                t_loss = self.compute_transition_loss(x)
            else:
                t_loss = tf.zeros_like(x[:, 0])
            if self.learn_laplacian:
                lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
                #print("lpl HF")
            else:
                lpl_loss = tf.zeros_like(x[:, 0])

            #omega = tf.expand_dims(y[:, -1], -1)
            #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
            total_loss = self.alpha[0]*lpl_loss +\
                self.alpha[1]*t_loss 
            # weight the loss.
            if sample_weight is not None:
                total_loss *= sample_weight
            total_loss_mean=tf.reduce_mean(total_loss)
        # Compute gradients
        gradients = tape.gradient(total_loss_mean, trainable_vars)
        # remove nans and gradient clipping from transition loss.
        gradients = [tf.where(tf.math.is_nan(g), 1e-8, g) for g in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, self.gclipping)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return metrics. NOTE: This interacts badly with any regular MSE
        # compiled loss. Make it so that only custom metrics are updated?
        # Return metrics. NOTE: This interacts badly with any regular MSE
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        return loss_dict

    def test_step(self, data):
        r"""Same as train_step without the outer gradient tape.
        Does *not* update the NN weights.

        NOTE:
            1. Computes the exaxt same losses as train_step
            
            2. Ricci loss val can be separately enabled with
                
                >>> model.learn_ricci_val = True
            
            3. Requires additional tracing.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # unpack data
        # if len(data) == 3:
        #     x, aux, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = data
        #x,sample_weight, pbs, invmets, sources = data.values()
        y = None
        y_pred=None
        x = data["X_val"]
        sample_weight = None
        pbs = data["val_pullbacks"]
        invmets = data["inv_mets_val"]
        sources = data["sources_val"]
        #print("validation happening")
        #y_pred = self(x)
        # add loss contributions
        if self.learn_transition:
            t_loss = self.compute_transition_loss(x)
        else:
            t_loss = tf.zeros_like(x[:, 0])
        if self.learn_laplacian:
            lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
        else:
            lpl_loss = tf.zeros_like(x[:, 0])

        #omega = tf.expand_dims(y[:, -1], -1)
        #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
        total_loss = self.alpha[0]*lpl_loss +\
            self.alpha[1]*t_loss 
        # weight the loss.
        if sample_weight is not None:
            total_loss *= sample_weight
        
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        return loss_dict

    def save(self, filepath, **kwargs):
        r"""Saves the underlying neural network to filepath.

        NOTE: 
            Currently does not save the whole custom model.

        Args:
            filepath (str): filepath
        """
        # TODO: save graph? What about Optimizer?
        # https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        self.model.save(filepath=filepath, **kwargs)


convertcomptoreal=lambda test: tf.concat([tf.math.real(test),tf.math.imag(test)],-1)
def compute_source_for_harmonicForm(pg,points,HYMmetric,harmonicform_jbar,invmetric,pullbacks):
    ncoords = tf.shape(points[0])[-1] // 2 
    pointstensor=tf.constant(points)
    with tf.GradientTape(persistent=False) as tape1:
        tape1.watch(pointstensor)
        cpoints=point_vec_to_complex(pointstensor)
        HNu=tf.einsum('x,xb->xb',tf.cast(HYMmetric(pointstensor),tf.complex64),tf.cast(harmonicform_jbar(tf.cast(cpoints,tf.complex64)),tf.complex64))#complexpoints vs noncomplex
        real_part = tf.math.real(HNu)
        imag_part = tf.math.imag(HNu)

        # Stack them along a new dimension
        Hnustack = tf.stack([real_part, imag_part], axis=1)# put at the 0 position
    dHnu = tape1.batch_jacobian(Hnustack, pointstensor)
    #dHnu = tape1.batch_jacobian(HNu, pointstensor)
    #dx_Hnu, dy_Hnu = \
    #    0.5*dHnu[:,:, :ncoords], \
    #    0.5*dHnu[:,:, ncoords:]
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the derivative index is the second index (the last index, as seen from above)
    dx_Hnu, dy_Hnu = \
        0.5*dHnu[:,:,:, :ncoords], \
        0.5*dHnu[:,:,:, ncoords:]
    # this should be holo derivative on the second index
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second d-index (the last index, as seen from above)
    dz_Hnu = tf.complex(dx_Hnu[:,0]+dy_Hnu[:,1],dx_Hnu[:,1]-dy_Hnu[:,0])
    #check - e.g.  Hnu might be x+iy = z. We want the derivative to be 1.
    # so take Hnu = R + iY
    # now do: dxR + dyI + i(dxI-dyR), gives 1/2 + 1/2  +0 + 0= 1
    # same for iz, now get 0 + 0 + i( 1/2 -(-1/2)) =i 
    return -tf.einsum('xba,xbj,xai,xji->x',invmetric,tf.math.conj(pullbacks),pullbacks,dz_Hnu) #the barred index is first, and the derivative index is second! Same is true, in fact, for the inverse metric

def prepare_dataset_HarmonicForm(point_gen, data,n_p, dirname, metricModel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,HYMmetric,val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True):
    r"""Prepares training and validation data from point_gen.

    Note:
        The dataset will be saved in `dirname/dataset.npz`.

    Args:
        point_gen (PointGenerator): Any point generator.
        n_p (int): # of points.
        dirname (str): dir name to save data.
        val_split (float, optional): train-val split. Defaults to 0.1.
        ltails (float, optional): Discarded % on the left tail of weight 
            distribution.
        rtails (float, optional): Discarded % on the left tail of weight 
            distribution.
        normalize_to_vol_j (bool, optional): Normalize such that

            .. math::
            
                \int_X \det(g) = \sum_p \det(g) * w|_p  = d^{ijk} t_i t_j t_k

            Defaults to True.

    Returns:
        np.float: kappa = vol_k / vol_cy
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    tf.keras.backend.clear_session()
    gc.collect()
    # new_np = int(round(n_p/(1-ltails-rtails)))
    # print("gets, generating")
    # pwo = point_gen.generate_point_weights(new_np, omega=True)
    # print("gets this far, then")
    # if len(pwo) < new_np:
    #     new_np = int((new_np-len(pwo))/len(pwo)*new_np + 100)
    #     pwo2 = point_gen.generate_point_weights(new_np, omega=True)
    #     pwo = np.concatenate([pwo, pwo2], axis=0)
    # print("gets this far, then2")
    # new_np = len(pwo)
    # sorted_weights = np.sort(pwo['weight'])
    # lower_bound = sorted_weights[round(ltails*new_np)]
    # upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    # mask = np.logical_and(pwo['weight'] >= lower_bound,
    #                       pwo['weight'] <= upper_bound)
    # weights = np.expand_dims(pwo['weight'][mask], -1)
    # omega = np.expand_dims(pwo['omega'][mask], -1)
    # omega = np.real(omega * np.conj(omega))
    
    # new_np = len(weights)
    # t_i = int((1-val_split)*new_np)
    # points = pwo['point'][mask]
    # print("gets this far, then2.1")

    # if normalize_to_vol_j:
    #     pbs = point_gen.pullbacks(points)
    #     fs_ref = point_gen.fubini_study_metrics(points, vol_js=np.ones_like(point_gen.kmoduli))
    #     fs_ref_pb = tf.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
    #     aux_weights = omega.flatten() / weights.flatten()
    #     norm_fac = point_gen.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / aux_weights)
    #     print("point_gen.vol_j_norm")
    #     print(point_gen.vol_j_norm)
    #     weights = norm_fac * weights # I.E. this is vol_j_norm/ integral of g_FS. That is, we normalise our volume to d_rst 1 1 1, when it is calculated with integral of omega wedge omegabar, i.e. just the weights. I.e. sum over just weights is that.

    # print("gets this far, then3")
    # X_train = np.concatenate((points[:t_i].real, points[:t_i].imag), axis=-1)
    # y_train = np.concatenate((weights[:t_i], omega[:t_i]), axis=1)
    # X_val = np.concatenate((points[t_i:].real, points[t_i:].imag), axis=-1)
    # y_val = np.concatenate((weights[t_i:], omega[t_i:]), axis=1)

    # pullbacks = point_gen.pullbacks(points)
    # train_pullbacks=tf.cast(pullbacks[:t_i],tf.complex64) 
    # val_pullbacks=tf.cast(pullbacks[t_i:],tf.complex64) 
    
    # realpoints=tf.concat((tf.math.real(points), tf.math.imag(points)), axis=-1)
    # realpoints=tf.cast(realpoints,tf.float32)

    # X_train=tf.cast(X_train,tf.float32)
    # y_train=tf.cast(y_train,tf.float32)
    # X_val=tf.cast(X_val,tf.float32)
    # y_val=tf.cast(y_val,tf.float32)
    # #realpoints=tf.cast(realpoints,tf.float32)
    
    # mets = metricModel(realpoints)
    # absdets = tf.abs(tf.linalg.det(mets))
    # inv_mets=tf.linalg.inv(mets)
    # inv_mets_train=inv_mets[:t_i]
    # inv_mets_val=inv_mets[t_i:]
    # print("gets this far, then4")

    X_train=tf.cast(data['X_train'],tf.float32)
    y_train=tf.cast(data['y_train'],tf.float32)
    X_val=tf.cast(data['X_val'],tf.float32)
    y_val=tf.cast(data['y_val'],tf.float32)
    ncoords=int(len(X_train[0])/2)

    #y_train=data['y_train']
    #y_val=data['y_val']
    ys=tf.concat((y_train,y_val),axis=0)
    weights=tf.cast(tf.expand_dims(ys[:,0],axis=-1),tf.float32)
    omega=tf.cast(tf.expand_dims(ys[:,1],axis=-1),tf.float32)

    realpoints=tf.concat((X_train,X_val),axis=0)
    points=tf.complex(realpoints[:,0:ncoords],realpoints[:,ncoords:])
    new_np = len(realpoints)
    t_i = int((1-val_split)*new_np)
    print('generating pullbacks')

    #still need to generate pullbacks apparently
    pullbacks = point_gen.pullbacks(points)
    print("TESTING METS")
    test = tf.cast(np.array([[-1.58031970e-01, 1.00000000e+00, 5.12138605e-01, 1.00000000e+00,
  1.00000000e+00, 2.60694236e-01, 1.00000000e+00, 4.38341200e-01,
  7.85521686e-01, 2.92491859e-17, 6.30328476e-01, 1.35924055e-17,
  -2.69683842e-17, -2.90005326e-01, 4.55345515e-18, -1.35843813e-01]]),tf.float32)
    print(metricModel.fubini_study_pb(test))
    print(metricModel(test))
    print(metricModel.BASIS['KMODULI'])
    print("TESTING PBS")
    print(realpoints[0])
    print(pullbacks[0])
    pullbacks_test_remove = metricModel.pullbacks(realpoints)
    print(pullbacks_test_remove[0])
    print("DONE TESTING PBS")
    train_pullbacks=tf.cast(pullbacks[:t_i],tf.complex64) 
    val_pullbacks=tf.cast(pullbacks[t_i:],tf.complex64) 

    print('generating mets')

        
    mets = batch_process_helper_func(metricModel, (realpoints,), batch_indices=(0,), batch_size=10000)
    
    absdets = tf.abs(tf.linalg.det(mets))
    inv_mets=tf.linalg.inv(mets)
    inv_mets_train=inv_mets[:t_i]
    inv_mets_val=inv_mets[t_i:]

    #linebundleindices=tf.convert_to_tensor(np.ones_like(kmoduli))
    #J_abbar = -ig_abbar
    #2jnp.pi*(kJ) is F, but J = -ig so it works out. Neecd to pull back
    #linebundleforHYM=np.array([-4,2,-2,-1])
    # F_forsource = 2j*np.pi*(-1j)*point_gen.fubini_study_metrics(points, vol_js=linebundleforHYM)
    # F_forsource_pb = tf.einsum('xai,xij,xbj->xab', pullbacks, F_forsource, np.conj(pullbacks))

    # FS_metric_pb = tf.einsum('xai,xij,xbj->xab', pullbacks, point_gen.fubini_study_metrics(points, vol_js=point_gen.kmoduli), np.conj(pullbacks))
    # FSmetricdets=tf.abs(tf.linalg.det(FS_metric_pb))
    # FSmetricinv = tf.linalg.inv(FS_metric_pb)
    # #print(tf.shape(fs_ref))
    
    
     
    print("gets this far, then4")
    print("check slope: isec, FSmetric, CYomega,CYdirect")
    print(tf.einsum('abc,a,b,c',BASIS["INTNUMS"],point_gen.kmoduli,point_gen.kmoduli,linebundleforHYM))
    #print(weights[:,0])
    #print(sources.shape)
    #print((weights*sources).shape)
    #g = phimodel(points)

    # use gamma series
    det = tf.math.real(absdets)  # * factorial / (2**nfold)
    #print("hi")
    det_over_omega = det / omega[:,0]
    #print("hi")
    volume_cy = tf.math.reduce_mean(weights[:,0], axis=-1)# according to raw CY omega calculation and sampling...
    #print("hi")
    #including the 6
    vol_k = tf.math.reduce_mean(det_over_omega * weights[:,0], axis=-1)
    #print("hi")
    kappaover6 = tf.cast(vol_k,tf.float32) / tf.cast(volume_cy,tf.float32)
    #rint(ratio)
    #print("hi")
    tf.cast(kappaover6,tf.float32)
    weightscomp=tf.cast(weights[:,0],tf.complex64)
    #print("hi")
    det = tf.cast(det,tf.float32)
    print('kappa over 6, returned as kappa: '+ str(kappaover6))
    #print("gets this far, then5")





    print('sourcesfor harmonic - generating')
    print(functionforbaseharmonicform_jbar.__name__)
    print("linebundleforHYM", linebundleforHYM)
    sourcesForHarmonic = batch_process_helper_func(
        compute_source_for_harmonicForm,
        (point_gen, realpoints, HYMmetric, functionforbaseharmonicform_jbar, inv_mets, tf.cast(pullbacks, tf.complex64)),
        batch_indices=(1, 4, 5),
        batch_size=10000
    )

    sources_train=tf.cast(sourcesForHarmonic[:t_i],tf.complex64)
    sources_val=tf.cast(sourcesForHarmonic[t_i:],tf.complex64)

   #test_realpoints = realpoints[::7]
    #test_inv_mets = inv_mets[::7]
    #test_pullbacks = pullbacks[::7]
    #test_sources = compute_source_for_harmonicForm(
    #    point_gen,
    #    test_realpoints,
    #    HYMmetric,
    #    functionforbaseharmonicform_jbar,
    #    test_inv_mets,
    #    tf.cast(test_pullbacks, tf.complex64))
    #batched_result_subset = sourcesForHarmonic[::7]
    #
    ## Check if they match
    #difference = tf.abs(test_sources - batched_result_subset)
    #max_diff = tf.reduce_max(difference)
    #print(f"Maximum difference between batched and unbatched computation: {max_diff}")
    #cast integral_weights
    cy_weights_for_vol_j_real= weights[:,0]*det/(6*omega[:,0])#why, because of the weird expand_dims?
    cy_weights_for_vol_j_complex = tf.cast(cy_weights_for_vol_j_real,tf.complex64)
    integrate_abs_sources=tf.reduce_mean(cy_weights_for_vol_j_real*tf.math.abs(sourcesForHarmonic))
    integrate_sources=tf.reduce_mean(cy_weights_for_vol_j_complex*sourcesForHarmonic)
    
    print("integrate absolute value of source: " + str(integrate_abs_sources))
    print("integrate source: " + str(integrate_sources))

    #print(norm_fac)
    #print(norm_fac)
    #slopefromvolCYrhoCY=(2/np.pi)*(1/(ratio))*tf.math.real(tf.reduce_mean(weightscomp*sourcesCY, axis=-1))
    # volfromCY=tf.math.real(tf.reduce_mean(weightscomp, axis=-1))*kappaover6
    # slopefromvolCYrhoCY=(1/6)*(2/np.pi)*tf.math.real(tf.reduce_mean(weightscomp*sourcesCY, axis=-1))*kappaover6
    # # slopeCY2=(2/np.pi)*tf.reduce_mean((omega.flatten() / weights.flatten())* weights.flatten() *sourcesCY, axis=-1)
    # print("CY")
    # print(volfromCY)
    # print(slopefromvolCYrhoCY)
    # print(slopeCY2)
    #xba as the inverse has bbar first then 
    #print(FSmetricinv.shape)
    #print(fs_forsource.shape)
    # sourceFS=(1/2)*tf.einsum('xba,xab->x',FSmetricinv,F_forsource_pb)
    # #print(FSmetricdets[0:3])
    # #slopefromvolFSrhoFS=(2/np.pi)*(1/(6*ratio))*tf.reduce_mean((weights[:,0]/det)* tf.cast(FSmetricdets,tf.float32) *sourceFS, axis=-1)
    # #print('reduce')
    # #print(tf.reduce_mean(tf.linalg.det(FS_metric_pb)))
    # #slopefromvolFSrhoFS=(2/np.pi)*tf.reduce_mean((weights[:,0]/omega[:,0])* tf.cast(FSmetricdets,tf.float32) *sourceFS, axis=-1)/vol_k #vol_k is the actual CY volume.
    # #slopefromvolFSrhoFS=(1/((3/2) * np.pi))*(2/np.pi)*(6*norm_fac*kappaover6)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,tf.float32)/omega[:,0])*sourceFS , axis=-1)#vol_k is the actual CY volume.
    # slopefromvolFSrhoFS=(1/6)*(2/np.pi)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,tf.float32)/omega[:,0])*sourceFS , axis=-1)#vol_k is the actual CY volume.
    # #volfromFSmetric=tf.reduce_mean((weights[:,0]/omega[:,0])* tf.cast(FSmetricdets,tf.float32) , axis=-1)/vol_k #vol_k is the actual CY volume.
    # #volfromFSmetric=(6*norm_fac*kappaover6)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,tf.float32)/omega[:,0]) , axis=-1) #vol_k is the actual CY volume.
    # volfromFSmetric=tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,tf.float32)/omega[:,0]) , axis=-1) #vol_k is the actual CY volume.
    # print('FS vol and slope')
    # print(volfromFSmetric)
    # print(slopefromvolFSrhoFS)
    #print(tf.reduce_mean(weights[:,0], axis=-1))
    
    # save everything to compressed dict.
    np.savez_compressed(os.path.join(dirname, 'dataset'),
                        X_train=X_train,
                        y_train=y_train,
                        train_pullbacks=train_pullbacks,
                        inv_mets_train=inv_mets_train,
                        sources_train=sources_train,
                        X_val=X_val,
                        y_val=y_val,
                        val_pullbacks=val_pullbacks,
                        inv_mets_val=inv_mets_val,
                        sources_val=sources_val
                        )
    print("Calculated kappa/6?")
    return kappaover6#point_gen.compute_kappa(points, weights, omega)


def train_modelHF(HFmodel, data_train, optimizer=None, epochs=50, batch_sizes=[64, 10000],
                verbose=1, custom_metrics=[], callbacks=[], sw=False):
    r"""Training loop for fixing the Kähler class. It consists of two 
    optimisation steps. 
        1. With a small batch size and volk loss disabled.
        2. With only MA and volk loss enabled and a large batchsize such that 
            the MC integral is a reasonable approximation and we don't lose 
            the MA progress from the first step.

    Args:
        HFmodel (cymetric.models.tfmodels): Any of the custom metric models.
        data (dict): numpy dictionary with keys 'X_train' and 'y_train'.
        optimizer (tfk.optimiser, optional): Any tf optimizer. Defaults to None.
            If None Adam is used with default hyperparameters.
        epochs (int, optional): # of training epochs. Every training sample will
            be iterated over twice per Epoch. Defaults to 50.
        batch_sizes (list, optional): batch sizes. Defaults to [64, 10000].
        verbose (int, optional): If > 0 prints epochs. Defaults to 1.
        custom_metrics (list, optional): List of tf metrics. Defaults to [].
        callbacks (list, optional): List of tf callbacks. Defaults to [].
        sw (bool, optional): If True, use integration weights as sample weights.
            Defaults to False.

    Returns:
        model, training_history
    """
    #from pympler import tracker
    training_history = {}
    hist1 = {}
    # hist1['opt'] = ['opt1' for _ in range(epochs)]
    hist2 = {}
    # hist2['opt'] = ['opt2' for _ in range(epochs)]
    learn_laplacian = HFmodel.learn_laplacian
    learn_transition = HFmodel.learn_transition
    if sw:
        sample_weights = data_train['y_train'][:, -2]
    else:
        sample_weights = None
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    #permint= tracker.SummaryTracker()
    #track= tracker.SummaryTracker()
    #from pympler import summary
    ######for epoch in range(epochs):
    for epoch in range(1):
        #print("internal")
        #print(permint.print_diff())
        #current_summary = track.create_summary()
        ## Print the entire memory summary
        #print(summary.print_(current_summary))

        batch_size = batch_sizes[0]
        HFmodel.learn_transition = learn_transition
        HFmodel.learn_laplacian = learn_laplacian
        #print("internal2")
        #print(permint.print_diff())
        HFmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        if verbose > 0:
            print("\nEpoch {:2d}/{:d}".format(epoch + 1, epochs))

        history = HFmodel.fit(
            data_train,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks, sample_weight=sample_weights
        )
        #print("internal2")
        #print(permint.print_diff())
        #current_summary = track.create_summary()
        #print(summary.print_(current_summary))


        #print('fit run')
        #tf.print('fit run')
        for k in history.history.keys():
            if k not in hist1.keys():
                hist1[k] = history.history[k]
            else:
                hist1[k] += history.history[k]
        #print('added hist')
        #tf.print('added hist')
        #print(hist1)
        #print(history)
        del history
        if tf.math.is_nan(hist1['loss'][-1]):
            break
        #if history.history['transition_loss'][-1]<10**(-8):
        #    break
        # batch_size = min(batch_sizes[1], len(data['X_train']))
        # HFmodel.learn_kaehler = tf.cast(False, dtype=tf.bool)
        # HFmodel.learn_transition = tf.cast(False, dtype=tf.bool)
        # HFmodel.learn_ricci = tf.cast(False, dtype=tf.bool)
        # HFmodel.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        # HFmodel.learn_volk = tf.cast(True, dtype=tf.bool)
        # HFmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        # history = HFmodel.fit(
        #     data['X_train'], data['y_train'],
        #     epochs=1, batch_size=batch_size, verbose=verbose,
        #     callbacks=callbacks, sample_weight=sample_weights
        # )
        # for k in history.history.keys():
        #     if k not in hist2.keys():
        #         hist2[k] = history.history[k]
        #     else:
        #         hist2[k] += history.history[k]
    # training_history['epochs'] = list(range(epochs)) + list(range(epochs))
    # for k in hist1.keys():
    #     training_history[k] = hist1[k] + hist2[k]
    #for k in set(list(hist1.keys()) + list(hist2.keys())):
    print("finished epoch loop")
    for k in set(list(hist1.keys())):
        #training_history[k] = hist2[k] if k in hist2 and max(hist2[k]) != 0 else hist1[k]
        training_history[k] = hist1[k]
    training_history['epochs'] = list(range(epochs))
    return HFmodel, training_history

