from hls4ml.templates.templates import Backend


memory_config_template = """
        dnnl::memory::dims {layer_name}_{memory_object}_dims = {{{dims}}};
        {memory_object_type} {layer_name}_{memory_object}_memory = dnnl::memory({{
                {{{layer_name}_{memory_object}_dims}},
                dnnl::memory::data_type::{data_type},
                dnnl::memory::format_tag::{format_tag}}},
                eng);
        
        auto {layer_name}_{memory_object}_md = dnnl::memory::desc({{ 
                {{{layer_name}_{memory_object}_dims}},
                dnnl::memory::data_type::{data_type},
                dnnl::memory::format_tag::any}});\n"""


dense_config_template = """
        dnnl::memory::dims {layer_name}_output_dims = {{{output_dims}}};
        auto {layer_name}_output_md = dnnl::memory::desc({{ 
                {{{layer_name}_output_dims}},
                dnnl::memory::data_type::{data_type},
                dnnl::memory::format_tag::any}});

        auto {layer_name}_desc = dnnl::inner_product_forward::desc(
                dnnl::prop_kind::forward_inference,
                {input_desc}, {layer_name}_weights_md, {layer_name}_bias_md, {layer_name}_output_md);
        
        auto {layer_name}_prim_desc = dnnl::inner_product_forward::primitive_desc({layer_name}_desc, eng);
        
        {memory_object_type} {layer_name}_memory = dnnl::memory({layer_name}_prim_desc.dst_desc(), eng);
        
        net.push_back(dnnl::inner_product_forward({layer_name}_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
                {{DNNL_ARG_WEIGHTS, {layer_name}_weights_memory}},
                {{DNNL_ARG_BIAS, {layer_name}_bias_memory}},
                {{DNNL_ARG_DST, {layer_name}_memory}}}});\n"""

eltwise_config_template = """
        auto {layer_name}_desc = dnnl::eltwise_forward::desc(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::eltwise_{type}, {input_desc},
                {alpha});
        
        auto {layer_name}_prim_desc = dnnl::eltwise_forward::primitive_desc({layer_name}_desc, eng);
        
        net.push_back(dnnl::eltwise_forward({layer_name}_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
                {{DNNL_ARG_DST, {output_memory}}}}});\n"""

softmax_config_template = """
        auto {layer_name}_desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
                {input_desc}, {axis});
                
        auto {layer_name}_prim_desc = dnnl::softmax_forward::primitive_desc({layer_name}_desc, eng);
        
        net.push_back(dnnl::softmax_forward({layer_name}_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
                {{DNNL_ARG_DST, {output_memory}}}}});"""

garnet_config_template = """
        // Reorder encoder output
        dnnl::memory::dims encoder_reoder_dims = {{1, {n_vertices}, {n_propagate}}};
        auto encoder_reorder_mem_desc = dnnl::memory::desc(encoder_reoder_dims,
                                                dnnl::memory::data_type::{data_type},
                                                dnnl::memory::format_tag::acb}});
        auto encoder_reorder_mem = dnnl::memory(encoder_reorder_mem_desc, eng);
        
        // -D^2
        auto garnet_pow2_desc = dnnl::eltwise_forward::desc(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::eltwise_pow, {distance_cal_mem_desc},
                -1.0, 2.0);
        
        auto garnet_pow2_prim_desc = dnnl::eltwise_forward::primitive_desc(garnet_pow2_desc, eng);
        
        net.push_back(dnnl::eltwise_forward(garnet_pow2_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {distance_cal_mem}}},
                {{DNNL_ARG_DST, {distance_cal_mem}}}}});
        

        // exp(-D^2)
        auto garnet_exp_desc = dnnl::eltwise_forward::desc(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::eltwise_exp, {distance_cal_mem_desc});
        
        auto garnet_exp_prim_desc = dnnl::eltwise_forward::primitive_desc(garnet_exp_desc, eng);
        
        net.push_back(dnnl::eltwise_forward(garnet_exp_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {distance_cal_mem}}},
                {{DNNL_ARG_DST, {distance_cal_mem}}}}});


        // aggregation
        std::vector<float> aggregation_bias_data({n_propagate}, 0.0);
        dnnl::memory::dims aggregation_bias_dims = {{1, 1, {n_propagate}}};
        auto aggregation_bias_md = dnnl::memory::desc(aggregation_bias_dims, 
                dnnl::memory::data_type::{data_type},
                dnnl::memory::format_tag::{format_tag});
        auto aggregation_bias_mem = dnnl::memory(aggregation_bias_md, eng);
        write_to_dnnl_memory(aggregation_bias_data.data(), aggregation_bias_mem);

        dnnl::memory::dims aggregation_output_dims = {{{batch_size}, {n_aggregators}, {n_propagate}}};
        auto aggregation_output_md = dnnl::memory::desc(aggregation_output_dims, 
                dnnl::memory::data_type::{data_type},
                dnnl::memory::format_tag::{format_tag});
        auto aggregation_output_mem = dnnl::memory(aggregation_output_md, eng);

        auto aggregation_desc = dnnl::matmul::desc({distance_cal_mem_desc}, {encoder_mem_desc}, aggregation_bias_md, aggregation_output_md);
        auto aggregation_primitive_desc = dnnl::matmul::primitive_desc(aggregation_desc, eng);

        net.push_back(dnnl::matmul({aggregation_primitive_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, {distance_cal_mem}}},
                {{DNNL_ARG_WEIGHTS, {encoder_mem}}},
                {{DNNL_ARG_BIAS, aggregation_bias_mem}},
                {{DNNL_ARG_DST, aggregation_output_mem}}}});

        // aggregation 1/Vmax

        auto aggregation_linear_desc = dnnl::eltwise_forward::desc(
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::eltwise_linear, aggregation_output_mem.get_desc(), {inv_Vmax}, 0.0);
        
        auto aggregation_linear_prim_desc = dnnl::eltwise_forward::primitive_desc(aggregation_linear_desc, eng);
        
        net.push_back(dnnl::eltwise_forward(aggregation_linear_prim_desc));
        net_args.push_back({{{{DNNL_ARG_SRC, aggregation_output_mem}},
                {{DNNL_ARG_DST, aggregation_output_mem}}}});

        //Output transformation

        
        \n"""

class OneAPI(Backend):
    def __init__(self):
        super(OneAPI, self).__init__('oneAPI')
        self.register_config_template('Memory', memory_config_template)
        self.register_config_template('Dense', dense_config_template)
        self.register_config_template('Activation', eltwise_config_template)
        self.register_config_template('Softmax', softmax_config_template)

    def register_config_template(self, name, config_template):
        self.config_templates[name] = config_template
