use darling::{
    FromDeriveInput,
    ast::{Data, Style},
};
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{DeriveInput, Index, WhereClause};

use crate::ir::{
    generate::{bounded_where_clause, runtime_enum_uses_slots, runtime_slot_name},
    parse::assign::{Assign, AssignVariant},
    paths::prelude_type,
};

impl ToTokens for Assign {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let assign = prelude_type("Assign");
        let scope = prelude_type("Scope");

        let name = &self.ident;
        let expand_name = self.name_expand.clone().unwrap_or_else(|| format_ident!("{name}Expand"));
        let (generics, generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.where_clause();

        let init_mut_body = match &self.data {
            Data::Enum(_) if self.runtime_variants.is_present() => {
                self.init_mut_body_runtime_enum()
            }
            Data::Enum(_) => self.init_mut_body_comptime_enum(),
            Data::Struct(_) => self.init_mut_body_struct(),
        };

        let assign_body = match &self.data {
            Data::Enum(_) if self.runtime_variants.is_present() => self.assign_body_runtime_enum(),
            Data::Enum(_) => self.assign_body_comptime_enum(),
            Data::Struct(_) => self.assign_body_struct(),
        };
        let join_body = self.init_mut_join_body();
        let payload_assign = match &self.data {
            Data::Enum(variants) if self.runtime_variants.is_present()
                && runtime_enum_uses_slots(variants.iter().map(|variant| {
                    (variant.fields.style == Style::Struct
                        || (variant.fields.style == Style::Tuple && variant.fields.is_empty()),
                        variant.fields.iter().map(|field| &field.ty).collect())
                })) =>
            {
                let payload = format_ident!("{}Payload", expand_name);
                let fields: Vec<_> = variants.iter().enumerate().flat_map(|(v, variant)| {
                    (0..variant.fields.len()).map(move |i| runtime_slot_name(v, i))
                }).collect();
                quote! {
                    impl #generics #assign for #payload #generic_names #where_clause {
                        fn expand_assign(&mut self, scope: &mut #scope, value: Self) {
                            #(#assign::expand_assign(&mut self.#fields, scope, value.#fields);)*
                        }
                        fn init_mut(&self, scope: &mut #scope) -> Self {
                            Self { #(#fields: #assign::init_mut(&self.#fields, scope),)* }
                        }
                        fn init_mut_join(&self, scope: &mut #scope, other: &Self) -> Self {
                            Self { #(#fields: #assign::init_mut_join(&self.#fields, scope, &other.#fields),)* }
                        }
                    }
                }
            }
            _ => quote![],
        };

        tokens.extend(quote! {
            #payload_assign
            impl #generics #assign for #expand_name #generic_names #where_clause {
                fn expand_assign(&mut self, scope: &mut #scope, value: Self) {
                    use #assign as _;
                    #assign_body
                }

                fn init_mut(&self, scope: &mut #scope) -> Self {
                    use #assign as _;
                    #init_mut_body
                }
                fn init_mut_join(&self, scope: &mut #scope, other: &Self) -> Self {
                    use #assign as _;
                    #join_body
                }
            }
        });
    }
}

impl Assign {
    fn init_mut_join_body(&self) -> TokenStream {
        match &self.data {
            Data::Struct(fields) => {
                let values = fields.fields.iter().enumerate().map(|(i, field)| {
                    let member = field.ident.as_ref().map(|ident| quote![#ident])
                        .unwrap_or_else(|| { let index = Index::from(i); quote![#index] });
                    let value = if field.comptime.is_present() {
                        quote![self.#member.clone()]
                    } else {
                        quote![self.#member.init_mut_join(scope, &other.#member)]
                    };
                    if fields.style == Style::Struct { quote![#member: #value] } else { value }
                });
                match fields.style {
                    Style::Struct => quote![Self { #(#values,)* }],
                    Style::Tuple => quote![Self(#(#values,)*)],
                    Style::Unit => quote![Self],
                }
            }
            Data::Enum(_) if self.runtime_variants.is_present() => quote![
                Self {
                    discriminant: self.discriminant.init_mut_join(scope, &other.discriminant),
                    value: self.value.init_mut_join(scope, &other.value),
                }
            ],
            Data::Enum(variants) => {
                let arms = variants.iter().map(|variant| {
                    let name = &variant.ident;
                    let left: Vec<_> = (0..variant.fields.len()).map(|i| format_ident!("left_{i}")).collect();
                    let right: Vec<_> = (0..variant.fields.len()).map(|i| format_ident!("right_{i}")).collect();
                    let values: Vec<_> = variant.fields.iter().enumerate().map(|(i, field)| {
                        let a = &left[i]; let b = &right[i];
                        if field.comptime.is_present() { quote![#a.clone()] }
                        else { quote![#a.init_mut_join(scope, #b)] }
                    }).collect();
                    match variant.fields.style {
                        Style::Tuple => quote![
                            (Self::#name(#(#left,)*), Self::#name(#(#right,)*)) => Self::#name(#(#values,)*)
                        ],
                        Style::Struct => {
                            let names: Vec<_> = variant.fields.iter().map(|field| field.ident.as_ref().unwrap()).collect();
                            quote![
                                (Self::#name { #(#names: #left,)* }, Self::#name { #(#names: #right,)* })
                                    => Self::#name { #(#names: #values,)* }
                            ]
                        }
                        Style::Unit => quote![(Self::#name, Self::#name) => Self::#name],
                    }
                });
                quote![match (self, other) { #(#arms,)* _ => panic!("Can't join mismatched enum variants") }]
            }
        }
    }

    fn assign_body_struct(&self) -> TokenStream {
        let struct_ = self.data.as_ref().take_struct().unwrap();

        let fields = struct_.fields.iter().enumerate().map(|(i, field)| {
            let index = Index::from(i);
            match &field.ident {
                Some(name) if field.comptime.is_present() => quote![self.#name = value.#name;],
                Some(name) => {
                    quote![self.#name.expand_assign(scope, value.#name);]
                }
                None if field.comptime.is_present() => quote![self.#index = value.#index;],
                None => quote![self.#index.expand_assign(scope, value.#index);],
            }
        });
        quote![#(#fields)*]
    }

    fn init_mut_body_struct(&self) -> TokenStream {
        let struct_ = self.data.as_ref().take_struct().unwrap();

        let fields = struct_.fields.iter().enumerate().map(|(i, field)| {
            let index = Index::from(i);
            match &field.ident {
                Some(name) if field.comptime.is_present() => quote![#name: self.#name.clone()],
                Some(name) => {
                    quote![#name: self.#name.init_mut(scope)]
                }
                None if field.comptime.is_present() => quote![self.#index.clone()],
                None => quote![self.#index.init_mut(scope)],
            }
        });

        match struct_.style {
            Style::Tuple => quote![Self(#(#fields,)*)],
            Style::Struct => quote![Self { #(#fields,)* }],
            Style::Unit => quote![Self],
        }
    }

    fn assign_body_comptime_enum(&self) -> TokenStream {
        let variants = self.data.as_ref().take_enum().unwrap();

        let branches = variants
            .iter()
            .map(|variant| self.assign_body_comptime_variant(variant));

        quote![match (self, value) {
            #(#branches,)*
            _ => core::panic!("Can't assign to mismatched enum variants"),
        }]
    }

    fn assign_body_comptime_variant(&self, variant: &AssignVariant) -> TokenStream {
        let variant_name = &variant.ident;

        match variant.fields.style {
            Style::Tuple => {
                let field_names_this =
                    (0..variant.fields.len()).map(|i| format_ident!("_{i}_this"));
                let field_names_other =
                    (0..variant.fields.len()).map(|i| format_ident!("_{i}_other"));

                let fields = variant.fields.iter().enumerate().map(|(i, field)| {
                    let name_this = format_ident!("_{i}_this");
                    let name_other = format_ident!("_{i}_other");
                    match field.comptime.is_present() {
                        true => quote![*#name_this = #name_other;],
                        false => quote![#name_this.expand_assign(scope, #name_other);],
                    }
                });

                quote! {
                    (Self::#variant_name(#(#field_names_this,)*),
                        Self::#variant_name(#(#field_names_other,)*)
                    ) => {
                        #(#fields)*
                    }
                }
            }
            Style::Struct => {
                let field_names_this = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let rename = format_ident!("{name}_this");
                    quote![#name: #rename]
                });
                let field_names_other = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let rename = format_ident!("{name}_other");
                    quote![#name: #rename]
                });

                let fields = variant.fields.iter().map(|field| {
                    let name = field.ident.as_ref().unwrap();
                    let name_this = format_ident!("{name}_this");
                    let name_other = format_ident!("{name}_other");
                    match field.comptime.is_present() {
                        true => quote![*#name_this = #name_other;],
                        false => quote![#name_this.expand_assign(scope, #name_other);],
                    }
                });

                quote! {
                    (Self::#variant_name { #(#field_names_this,)* },
                        Self::#variant_name { #(#field_names_other,)* }
                    ) => {
                        #(#fields)*
                    }
                }
            }
            Style::Unit => {
                quote![(Self::#variant_name, Self::#variant_name) => {}]
            }
        }
    }

    fn init_mut_body_comptime_enum(&self) -> TokenStream {
        let variants = self.data.as_ref().take_enum().unwrap();

        let branches = variants
            .iter()
            .map(|variant| self.init_mut_body_comptime_variant(variant));

        quote![match self {
            #(#branches,)*
        }]
    }

    fn init_mut_body_comptime_variant(&self, variant: &AssignVariant) -> TokenStream {
        let variant_name = &variant.ident;

        let field_names = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| field.ident.clone().unwrap_or_else(|| format_ident!("_{i}")));

        let fields = variant.fields.iter().enumerate().map(|(i, field)| {
            let index = format_ident!("_{i}");
            match &field.ident {
                Some(name) => {
                    quote![#name: #name.init_mut(scope)]
                }
                None => quote![#index.init_mut(scope)],
            }
        });

        match variant.fields.style {
            Style::Tuple => {
                quote![Self::#variant_name(#(#field_names,)*) => Self::#variant_name(#(#fields,)*)]
            }
            Style::Struct => {
                quote![Self::#variant_name { #(#field_names,)* } => Self::#variant_name{ #(#fields,)* }]
            }
            Style::Unit => quote![Self::#variant_name => Self::#variant_name],
        }
    }

    fn assign_body_runtime_enum(&self) -> TokenStream {
        quote! {
            self.discriminant.expand_assign(scope, value.discriminant);
            self.value.expand_assign(scope, value.value);
        }
    }

    fn init_mut_body_runtime_enum(&self) -> TokenStream {
        quote! {
            Self {
                discriminant: self.discriminant.init_mut(scope),
                value: self.value.init_mut(scope),
            }
        }
    }

    fn where_clause(&self) -> Option<WhereClause> {
        let ruda_type = prelude_type("RudaType");
        let assign = prelude_type("Assign");

        let fields: Vec<_> = match &self.data {
            Data::Enum(variants) => variants
                .iter()
                .flat_map(|it| it.fields.fields.iter())
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
            Data::Struct(fields) => fields
                .iter()
                .filter(|it| !it.comptime.is_present())
                .cloned()
                .collect(),
        };
        bounded_where_clause(
            &self.generics,
            fields,
            |param| quote!(<#param as #ruda_type>::ExpandType: #assign),
        )
    }
}

pub fn generate_ruda_type_mut(input: &DeriveInput) -> syn::Result<TokenStream> {
    let assign = Assign::from_derive_input(input)?;
    Ok(assign.into_token_stream())
}
