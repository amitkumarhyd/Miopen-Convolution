/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#ifndef GPU_AMD_MIOPEN_CONVOLUTION_HPP
#define GPU_AMD_MIOPEN_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/amd/miopen_convolution_impl.hpp"
#include "gpu/amd/miopen_convolution_pd.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_convolution_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_fwd_pd_t {
        using miopen_convolution_fwd_pd_t::miopen_convolution_fwd_pd_t;
        pd_t(const pd_t &other)
            : miopen_convolution_fwd_pd_t(other)
            , impl_(other.impl_)
            , dst_md_temp_(other.dst_md_temp_) {}

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm_t = primitive_attr_t::skip_mask_t;

            const auto attr_skip_mask = sm_t::oscale_runtime | sm_t::post_ops;

            bool ok = utils::one_of(desc()->prop_kind,
                    prop_kind::forward_training, prop_kind::forward_inference);
            ok = ok && attr()->has_default_values(attr_skip_mask);
            ok = ok && post_ops_ok(attr());
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                weights_md_.data_type, dst_md_.data_type)
                            || (utils::everyone_is(f16, src_md_.data_type,
                                    weights_md_.data_type,dst_md_.data_type))
                            || (utils::everyone_is(s8, src_md_.data_type,
                                        weights_md_.data_type)
                                    && utils::one_of(
                                            dst_md_.data_type, f32, s8)));
                           
            ok = ok && this->set_default_formats();
            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5 && src_md_.data_type != s8);
            ok = ok
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            src_md_.data_type == s8
                                    && attr()->output_scales_.mask_ == 0);
            ok = ok
                    && IMPLICATION(
                            src_md_.data_type == s8, check_s8_configuration())
                    // miopenOpTensor used for bias add requires both tensors
                    // to have the same data type.
                    && IMPLICATION(with_bias(),
                            dst_md_.data_type == bias_md_.data_type);
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&dst_md_);
            if (with_bias()) ok = ok && memory_format_ok(&bias_md_);
            
            std::cout<<"srcmd_format :"<<src_md_.format_kind<<std::endl;
            std::cout<<"dstmd_format :"<<dst_md_.format_kind<<std::endl;
            std::cout<<"weightmd_format :"<<weights_md_.format_kind<<std::endl;

            //ok = ok && check_format();
            //std::cout<<"check_format_ok :"<< check_format() <<std::endl;

            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            const bool use_temp_dst = attr()->post_ops_.len() > 0;
            if (use_temp_dst) {
                dst_md_temp_ = dst_md_;
                if (dst_md_.data_type == s8) { dst_md_temp_.data_type = f32; }
            }

            impl_.reset(new miopen_convolution_impl_fwd_t());
            return impl_->init(engine, this, use_temp_dst);
        }

        bool check_format() const {
            return (memory_desc_wrapper(src_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,format_tag::acb,
                            format_tag::abcd, format_tag::acdb)
                    && memory_desc_wrapper(weights_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,format_tag::acb,
                            format_tag::abcd, format_tag::acdb)
                    && memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,format_tag::acb,
                            format_tag::abcd, format_tag::acdb));
        }

        bool with_scratchpad() const {
            return impl_->with_scratchpad(); }

        std::shared_ptr<miopen_convolution_impl_base_t> impl_;
        memory_desc_t dst_md_temp_;

        bool use_temp_dst() const { 
            if (impl_.get()) return impl_->use_temp_dst();
            return false;
        }

    private:
        bool set_default_formats() {
            using namespace format_tag;

            if (src_md_.data_type == dnnl_s8) {
                std::cout << ".hpp "<<__LINE__<<" : "<<__func__<<std::endl;
                auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
                auto wei_tag = with_groups()
                        ? utils::pick(ndims() - 3, gowi, gohwi, godhwi)
                        : utils::pick(ndims() - 3, owi, ohwi, odhwi);
                    std::cout << ".hpp "<<"dat_tag= "<<dat_tag<<std::endl;
                    std::cout << ".hpp "<<"wei_tag= "<<wei_tag<<std::endl;
                return set_default_formats_common(dat_tag, wei_tag, dat_tag);
            } //else if(src_md_.data_type == dnnl_f16) {}
                else {
                std::cout << ".hpp "<<__LINE__<<" : "<<__func__<<std::endl;
                auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
                auto wei_tag = with_groups()
                        ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                        : utils::pick(ndims() - 3, oiw, oihw, oidhw);
                    std::cout << ".hpp "<<"dat_tag= "<<dat_tag<<std::endl;
                    std::cout << ".hpp "<<"wei_tag= "<<wei_tag<<std::endl;
                return set_default_formats_common(dat_tag, wei_tag, dat_tag);
            }
        }

        bool post_ops_ok(const primitive_attr_t *attr) const {
            const auto &p = attr->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
            auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

            switch (p.len()) {
                case 0: return true; // no post_ops
                case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
                case 2:
                    if (src_md_.data_type == dnnl_s8 && is_eltwise(0)
                            && is_sum(1))
                        return true;
                    return (is_sum(0) && is_eltwise(1));
                default: return false;
            }

            return false;
        }

        bool check_s8_configuration() const {
            const auto check_nhwc = [](const memory_desc_t &md,
                                            bool is_weights = false) {
                miopenTensorLayout_t fmt;
                get_format(&md, fmt, is_weights);
                return fmt == miopenTensorNHWC;
            };

            return check_nhwc(src_md_) && check_nhwc(dst_md_)
                    && check_nhwc(weights_md_, true)
                    && (src_md_.dims[1] % 4) == 0 && (dst_md_.dims[1] % 4) == 0
                    && ndims() < 5;
        }
    };

    status_t init_temp_dst(engine_t *engine) {
        auto sycl_engine = utils::downcast<sycl_hip_engine_t *>(engine);
        memory_storage_t *scratch_ptr = nullptr;
        auto wrap = memory_desc_wrapper(pd()->dst_md_temp_);
        CHECK(sycl_engine->create_memory_storage(
                &scratch_ptr, memory_flags_t::alloc, wrap.size(), nullptr));
        scratch_storage.reset(scratch_ptr);

        CHECK(sycl_engine->create_memory_storage(
                &scratch_ptr, memory_flags_t::alloc, wrap.size(), nullptr));
        scratch_storage_2.reset(scratch_ptr);

        return status::success;
    }

    virtual status_t init(engine_t *engine) override {
        const auto impl = pd()->impl_.get();
        if (impl && impl->use_temp_dst()) { init_temp_dst(engine); }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return status::success; }

        execute_convolution(ctx, pd()->with_bias(), pd()->with_scratchpad());

        return status::success;
    }

    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    ::sycl::buffer<uint8_t, 1> &buffer(memory_storage_t *mem_storage) const {
        return utils::downcast<impl::sycl::sycl_buffer_memory_storage_t *>(
                mem_storage)
                ->buffer();
    }
    const pd_t *pd() const { 
        return (const pd_t *)primitive_t::pd().get();
    }

    std::shared_ptr<memory_storage_t> scratch_storage;
    std::shared_ptr<memory_storage_t> scratch_storage_2;
};

struct miopen_convolution_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_bwd_data_pd_t {
        using miopen_convolution_bwd_data_pd_t::
                miopen_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_data;
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, diff_src_md_.data_type,
                                weights_md_.data_type, diff_dst_md_.data_type)
                            || utils::everyone_is(f16, diff_src_md_.data_type,
                                    weights_md_.data_type,
                                    diff_dst_md_.data_type));
            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&diff_src_md_);
            ok = ok && memory_format_ok(&weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&bias_md_);
                ok = ok && bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            impl_.reset(new miopen_convolution_impl_bwd_data_t());
            return impl_->init(engine, this);
        }

        std::shared_ptr<miopen_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
        bool support_bias() const override { return true; }
    };

    ~miopen_convolution_bwd_data_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) {  return status::success; }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_convolution_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public miopen_convolution_bwd_weights_pd_t {
        using miopen_convolution_bwd_weights_pd_t::
                miopen_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_convolution_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            bool ok = desc()->prop_kind == prop_kind::backward_weights;
            ok = ok && this->set_default_formats();
            ok = ok
                    && (utils::everyone_is(f32, src_md_.data_type,
                                diff_weights_md_.data_type,
                                diff_dst_md_.data_type)
                            || utils::everyone_is(f16, src_md_.data_type,
                                    diff_weights_md_.data_type,
                                    diff_dst_md_.data_type));
            ok = ok
                    && IMPLICATION(
                            desc()->alg_kind == dnnl_convolution_winograd,
                            ndims() < 5);
            ok = ok && memory_format_ok(&src_md_);
            ok = ok && memory_format_ok(&diff_weights_md_);
            ok = ok && memory_format_ok(&diff_dst_md_);
            if (with_bias()) {
                ok = ok && memory_format_ok(&diff_bias_md_);
                ok = ok && diff_bias_md_.data_type == diff_dst_md_.data_type;
            }
            if (!ok) return status::unimplemented;

            impl_.reset(new miopen_convolution_impl_bwd_weights_t());
            if (check_for_zero_dims()) { return impl_->init_zero_dims(this); };

            return impl_->init(engine, this);
        }

        std::shared_ptr<miopen_convolution_impl_base_t> impl_;

        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, ncw, nchw, ncdhw);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
        bool with_scratchpad() const { return impl_->with_scratchpad(); }
    };

    ~miopen_convolution_bwd_weights_t() {}
    status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->check_for_zero_dims()) { return execute_zero_dims(ctx); }
        return execute_convolution(
                ctx, pd()->with_bias(), pd()->with_scratchpad());
    }
    status_t execute_convolution(
            const exec_ctx_t &ctx, bool with_bias, bool with_scratchpad) const;
    status_t execute_zero_dims(const exec_ctx_t &ctx) const;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif