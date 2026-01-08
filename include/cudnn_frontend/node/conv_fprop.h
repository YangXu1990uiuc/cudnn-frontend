#pragma once

#include "../graph_helpers.h"
#include "../node_interface.h"

namespace cudnn_frontend::graph {
class ConvolutionNode : public NodeCRTP<ConvolutionNode> {
   public:
    Conv_fprop_attributes attributes;

    ConvolutionNode(Conv_fprop_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::CONVOLUTION;
    }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating Node Type::CONVOLUTION " << attributes.name);

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
        CUDNN_FE_LOG_LABEL_ENDL("INFO:     Inferrencing properties for conv node " << attributes.name);

        attributes.fill_from_context(context);

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto& X = attributes.inputs.find(Conv_fprop_attributes::input_names::X)->second;
        auto& W = attributes.inputs.find(Conv_fprop_attributes::input_names::W)->second;
        auto& Y = attributes.outputs.find(Conv_fprop_attributes::output_names::Y)->second;

        auto const x_tensor_dim = X->get_dim();
        auto const w_tensor_dim = W->get_dim();
        auto y_tensor_dim       = Y->get_dim();

        // Only infer dims and strides if user did not set them
        if (y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            auto const& pre_padding  = attributes.get_pre_padding();
            auto const& post_padding = attributes.get_post_padding();
            auto const& stride       = attributes.get_stride();
            auto const& dilation     = attributes.get_dilation();
            // N
            y_tensor_dim[0] = x_tensor_dim[0];
            // PQ
            for (size_t dim = 2; dim < x_tensor_dim.size(); ++dim) {
                y_tensor_dim[dim] = 1 + (x_tensor_dim[dim] - dilation[dim - 2] * (w_tensor_dim[dim] - 1) - 1 +
                                         pre_padding[dim - 2] + post_padding[dim - 2]) /
                                            stride[dim - 2];
            }
            // K
            y_tensor_dim[1] = w_tensor_dim[0];
            Y->set_dim(y_tensor_dim);
        }
        if (Y->get_stride().empty()) {
            auto const& Y_dim = Y->get_dim();
            // Default to NHWC
            auto const& stride_order = detail::generate_NHWC_stride_order(Y_dim.size());
            Y->set_stride(detail::generate_stride(Y_dim, stride_order));
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
        CUDNN_FE_LOG_LABEL("INFO: Building ConvolutionNode operations " << attributes.name << " ");

        // Create convolution descriptor by directly calling cuDNN backend API
        ConvDesc_v8 convolution_descriptor;
        int64_t const spatial_dim_count = attributes.get_pre_padding().size();

        CHECK_CUDNN_STATUS(
            convolution_descriptor.initialize_managed_backend_pointer(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR),
            "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnCreate Failed");

        // Set compute type
        cudnnDataType_t cudnn_data_type;
        CHECK_CUDNN_STATUS(detail::convert_to_cudnn_type(attributes.compute_data_type, cudnn_data_type),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 CUDNN_TYPE_DATA_TYPE,
                                                 1,
                                                 &cudnn_data_type),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_COMP_TYPE Failed");

        // Set convolution mode
        cudnnConvolutionMode_t mode = detail::convert_to_cudnn_type(attributes.math_mode);
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                                                 CUDNN_TYPE_CONVOLUTION_MODE,
                                                 1,
                                                 &mode),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_CONV_MODE Failed");

        // Set spatial dimensions
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                                                 CUDNN_TYPE_INT64,
                                                 1,
                                                 &spatial_dim_count),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS Failed");

        // Set pre-padding
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_pre_padding().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS Failed");

        // Set post-padding
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_post_padding().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_POST_PADDINGS Failed");

        // Set dilation
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_dilation().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_DILATIONS Failed");

        // Set strides
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_descriptor.get_raw_desc(),
                                                 CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                                 CUDNN_TYPE_INT64,
                                                 spatial_dim_count,
                                                 attributes.get_stride().data()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: SetAttribute CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES Failed");

        CHECK_CUDNN_STATUS(detail::finalize(convolution_descriptor.get_raw_desc()),
                          "CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR: cudnnFinalize Failed");
        CUDNN_FE_LOG_LABEL_ENDL(convolution_descriptor);

        // Create operation by directly calling cuDNN backend API
        Operation_v8 convolution_operation;

        CHECK_CUDNN_STATUS(
            convolution_operation.initialize_managed_backend_pointer(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR),
            "CUDNN_BACKEND_OPERATION: cudnnCreate Failed");

        // Set input tensor X
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(X, Conv_fprop_attributes::input_names::X);
        auto x_desc = tensors.at(X->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &x_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X Failed");

        // Set weight tensor W
        CUDNN_FE_VALIDATE_AND_ASSIGN_INPUT_TENSOR(W, Conv_fprop_attributes::input_names::W);
        auto w_desc = tensors.at(W->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &w_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W Failed");

        // Set output tensor Y
        CUDNN_FE_VALIDATE_AND_ASSIGN_OUTPUT_TENSOR(Y, Conv_fprop_attributes::output_names::Y);
        auto y_desc = tensors.at(Y->second->get_uid())->get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &y_desc),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y Failed");

        // Set convolution descriptor
        auto conv_desc_ptr = convolution_descriptor.get_raw_desc();
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                 1,
                                                 &conv_desc_ptr),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC Failed");

        // Set alpha and beta
        float alpha = 1.0f;
        float beta = 0.0f;
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                                 CUDNN_TYPE_FLOAT,
                                                 1,
                                                 &alpha),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA Failed");
        CHECK_CUDNN_STATUS(detail::set_attribute(convolution_operation.get_raw_desc(),
                                                 CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                                 CUDNN_TYPE_FLOAT,
                                                 1,
                                                 &beta),
                          "CUDNN_BACKEND_OPERATION: SetAttribute CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA Failed");

        CHECK_CUDNN_STATUS(detail::finalize(convolution_operation.get_raw_desc()),
                          "CUDNN_BACKEND_OPERATION: cudnnFinalize Failed");

        operations.push_back(std::make_shared<Operation_v8>(std::move(convolution_operation)));

        auto const& non_virtual_uids = attributes.get_non_virtual_uids();
        uids_involved_in_operations.insert(non_virtual_uids.begin(), non_virtual_uids.end());
        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
        j.update(R"({"tag": "CONV_FPROP"})"_json);
    }
#endif
};

}  // namespace cudnn_frontend::graph