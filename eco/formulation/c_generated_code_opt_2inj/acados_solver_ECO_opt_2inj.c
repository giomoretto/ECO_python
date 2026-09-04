/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "ECO_opt_2inj_model/ECO_opt_2inj_model.h"


#include "ECO_opt_2inj_constraints/ECO_opt_2inj_constraints.h"
#include "ECO_opt_2inj_cost/ECO_opt_2inj_cost.h"



#include "acados_solver_ECO_opt_2inj.h"

#define NX     ECO_OPT_2INJ_NX
#define NZ     ECO_OPT_2INJ_NZ
#define NU     ECO_OPT_2INJ_NU
#define NP     ECO_OPT_2INJ_NP
#define NP_GLOBAL     ECO_OPT_2INJ_NP_GLOBAL
#define NY0    ECO_OPT_2INJ_NY0
#define NY     ECO_OPT_2INJ_NY
#define NYN    ECO_OPT_2INJ_NYN

#define NBX    ECO_OPT_2INJ_NBX
#define NBX0   ECO_OPT_2INJ_NBX0
#define NBU    ECO_OPT_2INJ_NBU
#define NG     ECO_OPT_2INJ_NG
#define NBXN   ECO_OPT_2INJ_NBXN
#define NGN    ECO_OPT_2INJ_NGN

#define NH     ECO_OPT_2INJ_NH
#define NHN    ECO_OPT_2INJ_NHN
#define NH0    ECO_OPT_2INJ_NH0
#define NPHI   ECO_OPT_2INJ_NPHI
#define NPHIN  ECO_OPT_2INJ_NPHIN
#define NPHI0  ECO_OPT_2INJ_NPHI0
#define NR     ECO_OPT_2INJ_NR

#define NS     ECO_OPT_2INJ_NS
#define NS0    ECO_OPT_2INJ_NS0
#define NSN    ECO_OPT_2INJ_NSN

#define NSBX   ECO_OPT_2INJ_NSBX
#define NSBU   ECO_OPT_2INJ_NSBU
#define NSH0   ECO_OPT_2INJ_NSH0
#define NSH    ECO_OPT_2INJ_NSH
#define NSHN   ECO_OPT_2INJ_NSHN
#define NSG    ECO_OPT_2INJ_NSG
#define NSPHI0 ECO_OPT_2INJ_NSPHI0
#define NSPHI  ECO_OPT_2INJ_NSPHI
#define NSPHIN ECO_OPT_2INJ_NSPHIN
#define NSGN   ECO_OPT_2INJ_NSGN
#define NSBXN  ECO_OPT_2INJ_NSBXN



// ** solver data **

ECO_opt_2inj_solver_capsule * ECO_opt_2inj_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(ECO_opt_2inj_solver_capsule));
    ECO_opt_2inj_solver_capsule *capsule = (ECO_opt_2inj_solver_capsule *) capsule_mem;

    return capsule;
}


int ECO_opt_2inj_acados_free_capsule(ECO_opt_2inj_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int ECO_opt_2inj_acados_create(ECO_opt_2inj_solver_capsule* capsule)
{
    int N_shooting_intervals = ECO_OPT_2INJ_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return ECO_opt_2inj_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int ECO_opt_2inj_acados_update_time_steps(ECO_opt_2inj_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "ECO_opt_2inj_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for ECO_opt_2inj_acados_create: step 1
 */
void ECO_opt_2inj_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = EXTERNAL;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = EXTERNAL;

    nlp_solver_plan->nlp_cost[N] = EXTERNAL;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* ECO_opt_2inj_acados_create_setup_dimensions(ECO_opt_2inj_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 0;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for ECO_opt_2inj_acados_create: step 3
 */
void ECO_opt_2inj_acados_create_setup_functions(ECO_opt_2inj_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], ECO_opt_2inj_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], ECO_opt_2inj_constr_h_fun);
        }
        capsule->nl_constr_h_fun_jac_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac_hess[i], ECO_opt_2inj_constr_h_fun_jac_uxt_zt_hess);
        }
    
        // external cost
        MAP_CASADI_FNC(ext_cost_0_fun, ECO_opt_2inj_cost_ext_cost_0_fun);
        MAP_CASADI_FNC(ext_cost_0_fun_jac, ECO_opt_2inj_cost_ext_cost_0_fun_jac);
        MAP_CASADI_FNC(ext_cost_0_fun_jac_hess, ECO_opt_2inj_cost_ext_cost_0_fun_jac_hess);



    
        // explicit ode
        capsule->expl_vde_forw = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_forw[i], ECO_opt_2inj_expl_vde_forw);
        }

        

        capsule->expl_ode_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_ode_fun[i], ECO_opt_2inj_expl_ode_fun);
        }

        capsule->expl_vde_adj = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_adj[i], ECO_opt_2inj_expl_vde_adj);
        }
        capsule->expl_ode_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_ode_hess[i], ECO_opt_2inj_expl_ode_hess);
        }

    
        // external cost
        capsule->ext_cost_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun[i], ECO_opt_2inj_cost_ext_cost_fun);
        }

        capsule->ext_cost_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac[i], ECO_opt_2inj_cost_ext_cost_fun_jac);
        }

        capsule->ext_cost_fun_jac_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(ext_cost_fun_jac_hess[i], ECO_opt_2inj_cost_ext_cost_fun_jac_hess);
        }

        

        
    } // N > 0
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, ECO_opt_2inj_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, ECO_opt_2inj_constr_h_e_fun);
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac_hess, ECO_opt_2inj_constr_h_e_fun_jac_uxt_zt_hess);
    
    
    
    // external cost - function
    MAP_CASADI_FNC(ext_cost_e_fun, ECO_opt_2inj_cost_ext_cost_e_fun);

    // external cost - jacobian
    MAP_CASADI_FNC(ext_cost_e_fun_jac, ECO_opt_2inj_cost_ext_cost_e_fun_jac);

    // external cost - hessian
    MAP_CASADI_FNC(ext_cost_e_fun_jac_hess, ECO_opt_2inj_cost_ext_cost_e_fun_jac_hess);

    // external cost - jacobian wrt params
    

    

#undef MAP_CASADI_FNC
}


/**
 * Internal function for ECO_opt_2inj_acados_create: step 5
 */
void ECO_opt_2inj_acados_create_set_default_parameters(ECO_opt_2inj_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));
    p[0] = 400;

    for (int i = 0; i <= N; i++) {
        ECO_opt_2inj_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for ECO_opt_2inj_acados_create: step 5
 */
void ECO_opt_2inj_acados_setup_nlp_in(ECO_opt_2inj_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        ECO_opt_2inj_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double* time_steps = malloc(N*sizeof(double));
        time_steps[0] = 0.5;
        time_steps[1] = 0.5;
        time_steps[2] = 0.5;
        time_steps[3] = 0.5;
        time_steps[4] = 0.5;
        time_steps[5] = 0.5;
        time_steps[6] = 0.5;
        time_steps[7] = 0.5;
        time_steps[8] = 0.5;
        time_steps[9] = 0.5;
        time_steps[10] = 0.5;
        time_steps[11] = 0.5;
        time_steps[12] = 0.5;
        time_steps[13] = 0.5;
        time_steps[14] = 0.5;
        time_steps[15] = 0.5;
        time_steps[16] = 0.5;
        time_steps[17] = 0.5;
        time_steps[18] = 0.5;
        time_steps[19] = 0.5;
        time_steps[20] = 0.5;
        time_steps[21] = 0.5;
        time_steps[22] = 0.5;
        time_steps[23] = 0.5;
        time_steps[24] = 0.5;
        time_steps[25] = 0.5;
        time_steps[26] = 0.5;
        time_steps[27] = 0.5;
        time_steps[28] = 0.5;
        time_steps[29] = 0.5;
        time_steps[30] = 0.5;
        time_steps[31] = 0.5;
        time_steps[32] = 0.5;
        time_steps[33] = 0.5;
        time_steps[34] = 0.5;
        time_steps[35] = 0.5;
        time_steps[36] = 0.5;
        time_steps[37] = 0.5;
        time_steps[38] = 0.5;
        time_steps[39] = 0.5;
        time_steps[40] = 0.5;
        time_steps[41] = 0.5;
        time_steps[42] = 0.5;
        time_steps[43] = 0.5;
        time_steps[44] = 0.5;
        time_steps[45] = 0.5;
        time_steps[46] = 0.5;
        time_steps[47] = 0.5;
        time_steps[48] = 0.5;
        time_steps[49] = 0.5;
        time_steps[50] = 0.5;
        time_steps[51] = 0.5;
        time_steps[52] = 0.5;
        time_steps[53] = 0.5;
        time_steps[54] = 0.5;
        time_steps[55] = 0.5;
        time_steps[56] = 0.5;
        time_steps[57] = 0.5;
        time_steps[58] = 0.5;
        time_steps[59] = 0.5;
        time_steps[60] = 0.5;
        time_steps[61] = 0.5;
        time_steps[62] = 0.5;
        time_steps[63] = 0.5;
        time_steps[64] = 0.5;
        time_steps[65] = 0.5;
        time_steps[66] = 0.5;
        time_steps[67] = 0.5;
        time_steps[68] = 0.5;
        time_steps[69] = 0.5;
        time_steps[70] = 0.5;
        time_steps[71] = 0.5;
        time_steps[72] = 3;
        time_steps[73] = 3;
        time_steps[74] = 3;
        time_steps[75] = 3;
        time_steps[76] = 3;
        time_steps[77] = 3;
        time_steps[78] = 3;
        time_steps[79] = 3;
        time_steps[80] = 3;
        time_steps[81] = 3;
        time_steps[82] = 3;
        time_steps[83] = 3;
        time_steps[84] = 3;
        time_steps[85] = 3;
        time_steps[86] = 3;
        time_steps[87] = 3;
        time_steps[88] = 3;
        time_steps[89] = 3;
        time_steps[90] = 3;
        time_steps[91] = 3;
        time_steps[92] = 3;
        time_steps[93] = 3;
        time_steps[94] = 3;
        time_steps[95] = 3;
        time_steps[96] = 3;
        time_steps[97] = 3;
        time_steps[98] = 3;
        time_steps[99] = 3;
        time_steps[100] = 3;
        time_steps[101] = 3;
        time_steps[102] = 3;
        time_steps[103] = 3;
        time_steps[104] = 3;
        time_steps[105] = 3;
        time_steps[106] = 3;
        time_steps[107] = 3;
        time_steps[108] = 3;
        time_steps[109] = 3;
        time_steps[110] = 3;
        time_steps[111] = 3;
        time_steps[112] = 3;
        time_steps[113] = 3;
        time_steps[114] = 3;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_steps[i]);
        }
        free(time_steps);
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.5;
        cost_scaling[1] = 0.5;
        cost_scaling[2] = 0.5;
        cost_scaling[3] = 0.5;
        cost_scaling[4] = 0.5;
        cost_scaling[5] = 0.5;
        cost_scaling[6] = 0.5;
        cost_scaling[7] = 0.5;
        cost_scaling[8] = 0.5;
        cost_scaling[9] = 0.5;
        cost_scaling[10] = 0.5;
        cost_scaling[11] = 0.5;
        cost_scaling[12] = 0.5;
        cost_scaling[13] = 0.5;
        cost_scaling[14] = 0.5;
        cost_scaling[15] = 0.5;
        cost_scaling[16] = 0.5;
        cost_scaling[17] = 0.5;
        cost_scaling[18] = 0.5;
        cost_scaling[19] = 0.5;
        cost_scaling[20] = 0.5;
        cost_scaling[21] = 0.5;
        cost_scaling[22] = 0.5;
        cost_scaling[23] = 0.5;
        cost_scaling[24] = 0.5;
        cost_scaling[25] = 0.5;
        cost_scaling[26] = 0.5;
        cost_scaling[27] = 0.5;
        cost_scaling[28] = 0.5;
        cost_scaling[29] = 0.5;
        cost_scaling[30] = 0.5;
        cost_scaling[31] = 0.5;
        cost_scaling[32] = 0.5;
        cost_scaling[33] = 0.5;
        cost_scaling[34] = 0.5;
        cost_scaling[35] = 0.5;
        cost_scaling[36] = 0.5;
        cost_scaling[37] = 0.5;
        cost_scaling[38] = 0.5;
        cost_scaling[39] = 0.5;
        cost_scaling[40] = 0.5;
        cost_scaling[41] = 0.5;
        cost_scaling[42] = 0.5;
        cost_scaling[43] = 0.5;
        cost_scaling[44] = 0.5;
        cost_scaling[45] = 0.5;
        cost_scaling[46] = 0.5;
        cost_scaling[47] = 0.5;
        cost_scaling[48] = 0.5;
        cost_scaling[49] = 0.5;
        cost_scaling[50] = 0.5;
        cost_scaling[51] = 0.5;
        cost_scaling[52] = 0.5;
        cost_scaling[53] = 0.5;
        cost_scaling[54] = 0.5;
        cost_scaling[55] = 0.5;
        cost_scaling[56] = 0.5;
        cost_scaling[57] = 0.5;
        cost_scaling[58] = 0.5;
        cost_scaling[59] = 0.5;
        cost_scaling[60] = 0.5;
        cost_scaling[61] = 0.5;
        cost_scaling[62] = 0.5;
        cost_scaling[63] = 0.5;
        cost_scaling[64] = 0.5;
        cost_scaling[65] = 0.5;
        cost_scaling[66] = 0.5;
        cost_scaling[67] = 0.5;
        cost_scaling[68] = 0.5;
        cost_scaling[69] = 0.5;
        cost_scaling[70] = 0.5;
        cost_scaling[71] = 0.5;
        cost_scaling[72] = 3;
        cost_scaling[73] = 3;
        cost_scaling[74] = 3;
        cost_scaling[75] = 3;
        cost_scaling[76] = 3;
        cost_scaling[77] = 3;
        cost_scaling[78] = 3;
        cost_scaling[79] = 3;
        cost_scaling[80] = 3;
        cost_scaling[81] = 3;
        cost_scaling[82] = 3;
        cost_scaling[83] = 3;
        cost_scaling[84] = 3;
        cost_scaling[85] = 3;
        cost_scaling[86] = 3;
        cost_scaling[87] = 3;
        cost_scaling[88] = 3;
        cost_scaling[89] = 3;
        cost_scaling[90] = 3;
        cost_scaling[91] = 3;
        cost_scaling[92] = 3;
        cost_scaling[93] = 3;
        cost_scaling[94] = 3;
        cost_scaling[95] = 3;
        cost_scaling[96] = 3;
        cost_scaling[97] = 3;
        cost_scaling[98] = 3;
        cost_scaling[99] = 3;
        cost_scaling[100] = 3;
        cost_scaling[101] = 3;
        cost_scaling[102] = 3;
        cost_scaling[103] = 3;
        cost_scaling[104] = 3;
        cost_scaling[105] = 3;
        cost_scaling[106] = 3;
        cost_scaling[107] = 3;
        cost_scaling[108] = 3;
        cost_scaling[109] = 3;
        cost_scaling[110] = 3;
        cost_scaling[111] = 3;
        cost_scaling[112] = 3;
        cost_scaling[113] = 3;
        cost_scaling[114] = 3;
        cost_scaling[115] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->expl_vde_forw[i]);
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_adj", &capsule->expl_vde_adj[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_hess", &capsule->expl_ode_hess[i]);
    }

    /**** Cost ****/
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun", &capsule->ext_cost_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac", &capsule->ext_cost_0_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac_hess", &capsule->ext_cost_0_fun_jac_hess);
    
    
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
        
        
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);
    
    




    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 1000;
    Zu[0] = 20;
    zl[0] = 1000;
    zu[0] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);


    // slacks terminal
    double* zluemem = calloc(4*NSN, sizeof(double));
    double* Zl_e = zluemem+NSN*0;
    double* Zu_e = zluemem+NSN*1;
    double* zl_e = zluemem+NSN*2;
    double* zu_e = zluemem+NSN*3;

    // change only the non-zero elements:
    Zl_e[0] = 1000;
    Zu_e[0] = 10;
    zl_e[0] = 1000;
    zu_e[0] = 10;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
    free(zluemem);

    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[0] = -100;
    ubx0[0] = 100;
    lbx0[1] = -100;
    ubx0[1] = 100;
    lbx0[2] = -100;
    ubx0[2] = 100;
    lbx0[3] = -100;
    ubx0[3] = 100;
    lbx0[4] = -100;
    ubx0[4] = 100;
    lbx0[5] = -100;
    ubx0[5] = 100;
    lbx0[6] = -100;
    ubx0[6] = 100;
    lbx0[7] = -100;
    ubx0[7] = 100;
    lbx0[8] = -100;
    ubx0[8] = 100;
    lbx0[9] = -100;
    ubx0[9] = 100;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);












    /* constraints that are the same for initial and intermediate */






    /* Path constraints */



    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[0] = -100;
    lh[1] = -100;
    lh[2] = -100;
    uh[0] = 100;
    uh[1] = 100;
    uh[2] = 100;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                      "nl_constr_h_fun_jac_hess", &capsule->nl_constr_h_fun_jac_hess[i-1]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
        
        
    }
    free(luh);








    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    idxsh[0] = 1;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ush", ush);
    }
    free(idxsh);
    free(lush);



    /* terminal constraints */





    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*NHN, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + NHN;
    lh_e[0] = -100;
    lh_e[1] = -100;
    lh_e[2] = -100;
    lh_e[3] = -100;
    lh_e[4] = -100;
    uh_e[0] = 100;
    uh_e[1] = 100;
    uh_e[2] = 100;
    uh_e[3] = 100;
    uh_e[4] = 100;

    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac_hess",
                                  &capsule->nl_constr_h_e_fun_jac_hess);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "uh", uh_e);
    
    
    free(luh_e);



    /* terminal soft constraints */








    // set up soft bounds for nonlinear constraints
    int* idxsh_e = malloc(NSHN * sizeof(int));
    idxsh_e[0] = 2;
    double* lush_e = calloc(2*NSHN, sizeof(double));
    double* lsh_e = lush_e;
    double* ush_e = lush_e + NSHN;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ush", ush_e);
    free(idxsh_e);
    free(lush_e);





}


static void ECO_opt_2inj_acados_create_set_opts(ECO_opt_2inj_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/


    int nlp_solver_exact_hessian = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess", &nlp_solver_exact_hessian);

    int exact_hess_dyn = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_constr", &exact_hess_constr);

    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0.01;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 5;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    int log_dual_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_dual_step_norm", &log_dual_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    // set SQP specific options
    double nlp_solver_tol_stat = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 200;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    double adaptive_levenberg_marquardt_obj_scalar = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_obj_scalar", &adaptive_levenberg_marquardt_obj_scalar);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    // QP scaling
    double qpscaling_ub_max_abs_eig = 100000;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_ub_max_abs_eig", &qpscaling_ub_max_abs_eig);

    double qpscaling_lb_norm_inf_grad_obj = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_lb_norm_inf_grad_obj", &qpscaling_lb_norm_inf_grad_obj);

    qpscaling_scale_objective_type qpscaling_scale_objective = NO_OBJECTIVE_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_objective", &qpscaling_scale_objective);

    ocp_nlp_qpscaling_constraint_type qpscaling_scale_constraints = NO_CONSTRAINT_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_constraints", &qpscaling_scale_constraints);

    // NLP QP tol strategy
    ocp_nlp_qp_tol_strategy_t nlp_qp_tol_strategy = FIXED_QP_TOL;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_strategy", &nlp_qp_tol_strategy);

    double nlp_qp_tol_reduction_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_reduction_factor", &nlp_qp_tol_reduction_factor);

    double nlp_qp_tol_safety_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_safety_factor", &nlp_qp_tol_safety_factor);

    double nlp_qp_tol_min_stat = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_stat", &nlp_qp_tol_min_stat);

    double nlp_qp_tol_min_eq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_eq", &nlp_qp_tol_min_eq);

    double nlp_qp_tol_min_ineq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_ineq", &nlp_qp_tol_min_ineq);

    double nlp_qp_tol_min_comp = 0.00000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_comp", &nlp_qp_tol_min_comp);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
    ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);
}


/**
 * Internal function for ECO_opt_2inj_acados_create: step 7
 */
void ECO_opt_2inj_acados_set_nlp_out(ECO_opt_2inj_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[0] = -100;
    x0[1] = -100;
    x0[2] = -100;
    x0[3] = -100;
    x0[4] = -100;
    x0[5] = -100;
    x0[6] = -100;
    x0[7] = -100;
    x0[8] = -100;
    x0[9] = -100;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for ECO_opt_2inj_acados_create: step 9
 */
int ECO_opt_2inj_acados_create_precompute(ECO_opt_2inj_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int ECO_opt_2inj_acados_create_with_discretization(ECO_opt_2inj_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != ECO_OPT_2INJ_N && !new_time_steps) {
        fprintf(stderr, "ECO_opt_2inj_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, ECO_OPT_2INJ_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    ECO_opt_2inj_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = ECO_opt_2inj_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    ECO_opt_2inj_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    ECO_opt_2inj_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    ECO_opt_2inj_acados_create_setup_functions(capsule);
    ECO_opt_2inj_acados_setup_nlp_in(capsule, N, new_time_steps);
    ECO_opt_2inj_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = ECO_opt_2inj_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int ECO_opt_2inj_acados_update_qp_solver_cond_N(ECO_opt_2inj_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from ECO_opt_2inj_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = ECO_opt_2inj_acados_create_precompute(capsule);
    return status;
}


int ECO_opt_2inj_acados_reset(ECO_opt_2inj_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int ECO_opt_2inj_acados_update_params(ECO_opt_2inj_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 1;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int ECO_opt_2inj_acados_update_params_sparse(ECO_opt_2inj_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int ECO_opt_2inj_acados_set_p_global_and_precompute_dependencies(ECO_opt_2inj_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, ECO_opt_2inj_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int ECO_opt_2inj_acados_solve(ECO_opt_2inj_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int ECO_opt_2inj_acados_setup_qp_matrices_and_factorize(ECO_opt_2inj_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int ECO_opt_2inj_acados_free(ECO_opt_2inj_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->expl_vde_forw[i]);
        
        external_function_external_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_external_param_casadi_free(&capsule->expl_vde_adj[i]);
        external_function_external_param_casadi_free(&capsule->expl_ode_hess[i]);
    }
    free(capsule->expl_vde_adj);
    free(capsule->expl_vde_forw);
    
    free(capsule->expl_ode_fun);
    free(capsule->expl_ode_hess);

    // cost
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac_hess);
    
    
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->ext_cost_fun[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac_hess[i]);
        
        
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_e_fun_jac_hess);
    
    

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac_hess[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    free(capsule->nl_constr_h_fun_jac_hess);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun_jac_hess);



    return 0;
}


void ECO_opt_2inj_acados_print_stats(ECO_opt_2inj_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[3216];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

int ECO_opt_2inj_acados_custom_update(ECO_opt_2inj_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *ECO_opt_2inj_acados_get_nlp_in(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *ECO_opt_2inj_acados_get_nlp_out(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *ECO_opt_2inj_acados_get_sens_out(ECO_opt_2inj_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *ECO_opt_2inj_acados_get_nlp_solver(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *ECO_opt_2inj_acados_get_nlp_config(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_config; }
void *ECO_opt_2inj_acados_get_nlp_opts(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *ECO_opt_2inj_acados_get_nlp_dims(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *ECO_opt_2inj_acados_get_nlp_plan(ECO_opt_2inj_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
