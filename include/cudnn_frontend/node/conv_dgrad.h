#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {

class DgradNode : public NodeCRTP<DgradNode> {
   public:
    Conv_dgrad_attributes attributes;

    DgradNode(Conv_dgrad_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::DGRAD;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating Node Type::DGRAD " << attributes.name);

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_pre_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Pre padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_post_padding().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Post padding not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_stride().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv strides not set.");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            attributes.get_dilation().empty(), error_code_t::ATTRIBUTE_NOT_SET, "Conv dilation not set.");

        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for dgrad node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, DY) -> DW works today.
        auto DX = attributes.outputs.find(Conv_dgrad_attributes::output_names::DX)->second;
        auto W  = attributes.inputs.find(Conv_dgrad_attributes::input_names::W)->second;
        auto DY = attributes.inputs.find(Conv_dgrad_attributes::input_names::DY)->second;

        auto const w_tensor_dim  = W->get_dim();
        auto const dy_tensor_dim = DY->get_dim();
        auto dx_tensor_dim       = DX->get_dim();

        RETURN_CUDNN_FRONTEND_ERROR_IF(DX->get_dim().empty(),
                                       error_code_t::ATTRIBUTE_NOT_SET,
                                       "For dgrad node, output dimension inferencing is not possible.");

        // No dim inferencing as inverse mapping from DY, W to DX is not unique.
        // Only infer strides if user did not set them
        if (DX->get_stride().empty()) {
            auto const& DX_dim = DX->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(DX_dim.size());
            DX->set_stride(detail::generate_stride(DX_dim, stride_order));
        }

        return {error_code_t::OK, ""};
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<Tensor_attributes::uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
        CUDNN_FRONTEND_UNUSED(raw_operations);
        CUDNN_FE_LOG_LABEL("INFO: Building DgradNode operations " << attributes.name << " ");

        // Create dgrad descriptor by directly calling cuDNN backend API
        ConvDesc_v8 dgrad_descriptor;
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();

        CHECK_CUDNN_STATUS(
            dgrad_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR),
            "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnCreate Failed");

        cudnnDataType_t cudnn_data_type;
        CHECK_CUDNN_STATUS(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 CUDNN_TYPE_DATA_TYPE,
                                                 1,
                                                 &cudnn_data_type),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");

        cudnnConvolutionMode_t mode = detail::convert_to_cudnn_type(attributes.math_mode);
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                                 CUDNN_TYPE_CONVOLUTION_MODE,
                                                 1,
                                                 &mode),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_CONV_MODE Failed");

        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                                 CUDNN_TYPE_INT64,
                                                 1,
                                                 &spatial_dim_count),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS Failed");

        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_pre_padding().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS Failed");

        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_post_padding().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_POST_PADDINGS Failed");

        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_dilation().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_DILATIONS Failed");

        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_stride().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES Failed");

        CHECK_CUDNN_STATUS(detail::finalize(dgrad_descriptor.get_raw_desc()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnFinalize Failed");
        CUDNN_FE_LOG_LABEL_ENDL(dgrad_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 dgrad_operation;

        CHECK_CUDNN_STATUS(
            dgrad_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR),
            "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");

        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(DX, Conv_dgrad_attributes::output_names::DX);
        auto dx_desc = tensors.at(DX->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &dx_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX Failed");

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(W, Conv_dgrad_attributes::input_names::W);
        auto w_desc = tensors.at(W->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &w_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W Failed");

        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(DY, Conv_dgrad_attributes::input_names::DY);
        auto dy_desc = tensors.at(DY->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &dy_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY Failed");

        auto conv_desc_ptr = dgrad_descriptor.get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &conv_desc_ptr),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC Failed");

        float alpha = 1.0f;
        float beta = 0.0f;
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                                 CUDNN_TYPE_FLOAT,
                                                 1,
                                                 &alpha),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA Failed");
        CHECK_CUDNN_STATUS(detail::set_attribute(dgrad_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                                 CUDNN_TYPE_FLOAT,
                                                 1,
                                                 &beta),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA Failed");

        CHECK_CUDNN_STATUS(detail::finalize(dgrad_operation.get_raw_desc()),
                          "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");

        operations.push_back(std::make_shared<Operation_v8>(std::move(dgrad_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"( {"tag": "CONV_DGRAD"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph