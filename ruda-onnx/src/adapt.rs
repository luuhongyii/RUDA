use std::collections::HashSet;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_quote, visit_mut::{self, VisitMut}};

/// Lower the fixed ONNX generator's tensor/module types to RUDA's backend-generic API.
pub(crate) fn model(tokens: TokenStream) -> TokenStream {
    let mut file: syn::File = syn::parse2(tokens).expect("generated model must be valid Rust");
    let modules = file.items.iter().filter_map(|item| match item {
        syn::Item::Struct(s) if s.ident == "Model" || s.ident.to_string().starts_with("Submodule") => {
            Some(s.ident.to_string())
        }
        _ => None,
    }).collect();
    Native { modules }.visit_file_mut(&mut file);
    file.into_token_stream()
}

struct Native { modules: HashSet<String> }

impl Native {
    fn backend_type(&self, name: &str) -> bool {
        self.modules.contains(name) || matches!(name,
            "Tensor" | "Linear" | "Conv1d" | "Conv2d" | "Conv3d" |
            "ConvTranspose1d" | "ConvTranspose2d" | "ConvTranspose3d" | "DeformConv2d" |
            "BatchNorm" | "InstanceNorm" | "LayerNorm" | "GroupNorm" | "PRelu" |
            "Lstm" | "BiLstm" | "CoupledLstm" | "CoupledBiLstm" | "LstmState" | "Gru" | "BiGru" | "Rnn" | "BiRnn" | "RnnState")
    }

    fn generic_path(&self, path: &mut syn::Path, expression: bool) {
        for segment in &mut path.segments {
            if self.backend_type(&segment.ident.to_string()) {
                match &mut segment.arguments {
                    syn::PathArguments::AngleBracketed(args) => {
                        if !args.args.first().is_some_and(|a| a.to_token_stream().to_string() == "B") {
                            args.args.insert(0, parse_quote!(B));
                        }
                    }
                    syn::PathArguments::None if !expression => {
                        segment.arguments = syn::PathArguments::AngleBracketed(parse_quote!(<B>));
                    }
                    _ => (),
                }
            }
        }
    }
}

impl VisitMut for Native {
    fn visit_macro_mut(&mut self, node: &mut syn::Macro) {
        use syn::parse::Parser;
        let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(mut values) = parser.parse2(node.tokens.clone()) {
            for value in &mut values { self.visit_expr_mut(value); }
            node.tokens = values.into_token_stream();
        } else if node.path.is_ident("vec") {
            let tokens = &node.tokens;
            if let Ok(mut array) = syn::parse2::<syn::Expr>(quote!([#tokens])) {
                self.visit_expr_mut(&mut array);
                if let syn::Expr::Repeat(repeat) = array {
                    let value = &repeat.expr;
                    let length = &repeat.len;
                    node.tokens = quote!(#value; #length);
                }
            }
        }
    }

    fn visit_item_struct_mut(&mut self, item: &mut syn::ItemStruct) {
        if self.modules.contains(&item.ident.to_string()) {
            item.generics.params.insert(0, parse_quote!(B: Backend));
            if let syn::Fields::Named(fields) = &mut item.fields {
                fields.named.insert(0, parse_quote!(__ruda_backend: core::marker::PhantomData<B>));
            }
        }
        visit_mut::visit_item_struct_mut(self, item);
    }

    fn visit_item_impl_mut(&mut self, item: &mut syn::ItemImpl) {
        if let syn::Type::Path(ty) = item.self_ty.as_ref()
            && ty.path.segments.last().is_some_and(|s| self.modules.contains(&s.ident.to_string())) {
            item.generics.params.insert(0, parse_quote!(B: Backend));
        }
        visit_mut::visit_item_impl_mut(self, item);
    }

    fn visit_path_mut(&mut self, path: &mut syn::Path) {
        let first = path.segments.first().map(|s| s.ident.to_string());
        if first.as_deref() == Some("burn_tensor") {
            let tail = path.segments.iter().skip(1).cloned().collect::<syn::punctuated::Punctuated<_, syn::Token![::]>>();
            *path = syn::parse2(quote!(ruda_tensor::api::#tail)).unwrap();
        } else if first.as_deref() == Some("Device") && path.segments.len() == 1 {
            *path = parse_quote!(B::Device);
        }
        visit_mut::visit_path_mut(self, path);
    }

    fn visit_use_tree_mut(&mut self, tree: &mut syn::UseTree) {
        if let syn::UseTree::Path(path) = tree && path.ident == "burn_tensor" {
            let tail = &path.tree;
            *tree = syn::parse2(quote!(ruda_tensor::api::#tail)).unwrap();
        }
        visit_mut::visit_use_tree_mut(self, tree);
    }

    fn visit_type_path_mut(&mut self, ty: &mut syn::TypePath) {
        visit_mut::visit_type_path_mut(self, ty);
        self.generic_path(&mut ty.path, false);
    }

    fn visit_expr_path_mut(&mut self, expr: &mut syn::ExprPath) {
        visit_mut::visit_expr_path_mut(self, expr);
        self.generic_path(&mut expr.path, true);
        if expr.path.segments.first().is_some_and(|s| s.ident == "ruda_tensor")
            && let Some(last) = expr.path.segments.last_mut()
            && let syn::PathArguments::AngleBracketed(args) = &mut last.arguments
            && matches!(last.ident.to_string().as_str(), "det" | "lp_norm" | "lp_norm_dims" | "stft" | "cfft") {
            args.args.insert(0, parse_quote!(B));
        }
    }

    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        visit_mut::visit_expr_mut(self, expr);
        if let syn::Expr::MethodCall(call) = expr && call.method == "try_into_vec" {
            call.method = syn::Ident::new("into_vec", call.method.span());
        }
        if let syn::Expr::Struct(value) = expr
            && value.path.is_ident("Self")
            && value.fields.iter().any(|f| matches!(&f.member, syn::Member::Named(n) if n == "device")) {
            value.fields.push(parse_quote!(__ruda_backend: core::marker::PhantomData));
        }
        if let syn::Expr::MethodCall(call) = expr && call.method == "supports_dtype" {
            let device = &call.receiver;
            let dtype = call.args.first().expect("device dtype query");
            *expr = parse_quote!(B::supports_dtype(&(#device), #dtype));
        }
        if let syn::Expr::MethodCall(call) = expr && call.method == "squeeze_dims" {
            let axes = call.args.first().expect("squeeze axes");
            call.args = parse_quote!(&((#axes).iter().map(|&axis| axis as isize).collect::<alloc::vec::Vec<_>>()));
        }
        if let syn::Expr::MethodCall(call) = expr
            && call.method == "into_scalar"
            && let Some(args) = &call.turbofish {
            let element = args.args.first().expect("scalar element type");
            let tensor = &call.receiver;
            *expr = parse_quote!({
                let data = (#tensor).into_data().convert::<#element>();
                let values = data.into_vec::<#element>().expect("ONNX scalar element type");
                assert_eq!(values.len(), 1, "ONNX scalar must contain one element");
                values[0]
            });
        }
    }
}
