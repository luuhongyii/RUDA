use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_quote, Data, DeriveInput, Fields};

use super::item::codegen_enum_full::contains_self;

pub(crate) fn generate(ast: &DeriveInput, paths: &crate::DerivePaths) -> TokenStream {
    let model = &paths.model;
    let name = &ast.ident;
    let mut generics = ast.generics.clone();
    if !generics.type_params().any(|param| param.ident == "B") {
        generics.params.push(parse_quote!(B: #model::tensor::backend::Backend));
    }
    let fields: Vec<_> = match &ast.data {
        Data::Struct(data) => data.fields.iter().collect(),
        Data::Enum(data) => data.variants.iter().flat_map(|variant| variant.fields.iter()).collect(),
        Data::Union(_) => unreachable!(),
    };
    for field in fields {
        if contains_self(&field.ty, name) { continue; }
        let ty = &field.ty;
        generics.make_where_clause().predicates.push(parse_quote!(#ty: #model::record::RuntimeRecord<B>));
    }
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let (_, ty_generics, _) = ast.generics.split_for_impl();
    let (encode, decode) = match &ast.data {
        Data::Struct(data) => {
            let (pattern, encoded, decoded) = fields_codec(&data.fields, paths);
            (quote! {
                let Self #pattern = self;
                #model::record::RuntimeRecordValue::Fields(0, [#(#encoded),*].into_iter().collect())
            }, {
                let count = data.fields.len();
                quote! {
                    let mut __ruda_fields = __ruda_value.into_fields(0, #count);
                    Self #decoded
                }
            })
        }
        Data::Enum(data) => {
            let mut encoders = Vec::new();
            let mut decoders = Vec::new();
            for (index, variant) in data.variants.iter().enumerate() {
                let tag = index as u32;
                let variant_name = &variant.ident;
                let count = variant.fields.len();
                let (pattern, encoded, decoded) = fields_codec(&variant.fields, paths);
                encoders.push(quote! {
                    Self::#variant_name #pattern => #model::record::RuntimeRecordValue::Fields(
                        #tag, [#(#encoded),*].into_iter().collect()),
                });
                decoders.push(quote! {
                    #tag => {
                        assert_eq!(__ruda_values.len(), #count, "Record variant field count mismatch");
                        let mut __ruda_fields = __ruda_values.into_iter();
                        Self::#variant_name #decoded
                    }
                });
            }
            (quote!(match self { #(#encoders)* }), quote! {
                let (__ruda_tag, __ruda_values) = __ruda_value.into_variant();
                match __ruda_tag { #(#decoders)* _ => panic!("Invalid record variant") }
            })
        }
        Data::Union(_) => unreachable!(),
    };
    quote! {
        impl #impl_generics #model::record::RuntimeRecord<B> for #name #ty_generics #where_clause {
            fn into_runtime(self, __ruda_precision: #model::record::RuntimePrecision) -> #model::record::RuntimeRecordValue {
                #encode
            }
            fn from_runtime(__ruda_value: #model::record::RuntimeRecordValue, __ruda_device: &B::Device) -> Self {
                #decode
            }
        }
    }
}

fn fields_codec(fields: &Fields, paths: &crate::DerivePaths) -> (TokenStream, Vec<TokenStream>, TokenStream) {
    let model = &paths.model;
    let bindings: Vec<_> = fields.iter().enumerate().map(|(index, _)| format_ident!("__ruda_field_{index}")).collect();
    let encoded = bindings.iter().map(|binding|
        quote!(#model::record::RuntimeRecord::<B>::into_runtime(#binding, __ruda_precision))).collect();
    let decoded: Vec<_> = fields.iter().map(|field| {
        let ty = &field.ty;
        quote!(<#ty as #model::record::RuntimeRecord<B>>::from_runtime(__ruda_fields.next().unwrap(), __ruda_device))
    }).collect();
    match fields {
        Fields::Unit => (quote!(), encoded, quote!()),
        Fields::Unnamed(_) => (quote!((#(#bindings),*)), encoded, quote!((#(#decoded),*))),
        Fields::Named(named) => {
            let names: Vec<_> = named.named.iter().map(|field| field.ident.as_ref().unwrap()).collect();
            (quote!({ #(#names: #bindings),* }), encoded, quote!({ #(#names: #decoded),* }))
        }
    }
}
