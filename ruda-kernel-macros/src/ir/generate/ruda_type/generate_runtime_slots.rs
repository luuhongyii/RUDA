use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, Index, Member};

use crate::ir::{
    generate::{runtime_enum_uses_slots, runtime_field_accessor, runtime_reverse_accessor, runtime_slot_name},
    parse::ruda_type::{RudaTypeEnum, RudaTypeVariant},
    paths::{core_path, prelude_type},
};

impl RudaTypeEnum {
    pub(super) fn uses_runtime_slots(&self) -> bool {
        runtime_enum_uses_slots(self.variants.iter().map(|variant| {
            (matches!(variant.fields, syn::Fields::Named(_))
                || matches!(&variant.fields, syn::Fields::Unnamed(fields) if fields.unnamed.is_empty()),
                variant.fields.iter().map(|field| &field.ty).collect())
        }))
    }

    pub(super) fn runtime_field_accessors(&self, slots: bool) -> TokenStream {
        let ruda_type = prelude_type("RudaType");
        let name = &self.name_expand;
        let (generics, names, clause) = self.generics.split_for_impl();
        let accessors = self.variants.iter().enumerate().flat_map(|(v, variant)| {
            let ruda_type = &ruda_type;
            variant.fields.iter().enumerate().map(move |(i, field)| {
                let member = field.ident.clone().map(Member::Named)
                    .unwrap_or_else(|| Member::Unnamed(Index::from(i)));
                let accessor = runtime_field_accessor(&variant.ident, &member);
                let ty = &field.ty;
                let field = runtime_slot_name(v, i);
                let value = if slots { quote![self.value.#field.clone()] }
                    else { quote![self.value.clone()] };
                let reverse = runtime_reverse_accessor(&variant.ident, variant.fields.len() - i - 1);
                quote! {
                    pub fn #accessor(&self) -> <#ty as #ruda_type>::ExpandType { #value }
                    pub fn #reverse(&self) -> <#ty as #ruda_type>::ExpandType { #value }
                }
            })
        });
        quote! {
            #[allow(non_snake_case)]
            impl #generics #name #names #clause { #(#accessors)* }
        }
    }

    pub(super) fn comptime_field_accessors(&self) -> TokenStream {
        let ruda_type = prelude_type("RudaType");
        let name = &self.name_expand;
        let (generics, names, clause) = self.generics.split_for_impl();
        let accessors = self.variants.iter().flat_map(|variant| {
            let ruda_type = &ruda_type;
            variant.fields.iter().enumerate().map(move |(i, field)| {
                let member = field.ident.clone().map(Member::Named)
                    .unwrap_or_else(|| Member::Unnamed(Index::from(i)));
                let accessor = runtime_field_accessor(&variant.ident, &member);
                let reverse = runtime_reverse_accessor(&variant.ident, variant.fields.len() - i - 1);
                let ty = &field.ty;
                let pattern = variant.runtime_pattern(name);
                let arg = &variant.field_names[i];
                quote! {
                    pub fn #accessor(&self) -> <#ty as #ruda_type>::ExpandType {
                        match self { #pattern => #arg.clone(), _ => unreachable!() }
                    }
                    pub fn #reverse(&self) -> <#ty as #ruda_type>::ExpandType { self.#accessor() }
                }
            })
        });
        quote! {
            #[allow(non_snake_case, unused)]
            impl #generics #name #names #clause { #(#accessors)* }
        }
    }

    pub(super) fn generate_runtime_slots(&self, with_launch: bool) -> TokenStream {
        if with_launch { return self.generate_runtime_slots_launch(); }
        let ruda_type = prelude_type("RudaType");
        let ruda_enum = prelude_type("RudaEnum");
        let into_mut = prelude_type("IntoMut");
        let scope = prelude_type("Scope");
        let debug = prelude_type("RudaDebug");
        let native = prelude_type("NativeExpand");
        let core = core_path();
        let name = &self.ident;
        let expand = &self.name_expand;
        let payload = format_ident!("{}Payload", expand);
        let vis = &self.vis;
        let (generics, names, clause) = self.generics.split_for_impl();
        let turbofish = names.as_turbofish();
        let slots: Vec<_> = self.variants.iter().enumerate().flat_map(|(v, variant)| {
            variant.fields.iter().enumerate().map(move |(i, field)| (v, i, runtime_slot_name(v, i), &field.ty))
        }).collect();
        let fields = slots.iter().map(|(_, _, field, ty)| quote![#field: <#ty as #ruda_type>::ExpandType]);
        let clones = slots.iter().map(|(_, _, field, _)| quote![#field: self.#field.clone()]);
        let mutable = slots.iter().map(|(_, _, field, _)| quote![#field: #into_mut::into_mut(self.#field, scope)]);
        let snapshots = slots.iter().map(|(_, _, field, _)| quote![#field: #into_mut::expand_reference_value(self.#field, scope)]);
        let inactive = slots.iter().map(|(_, _, field, ty)| quote![#field: <#ty as #ruda_type>::__expand_inactive_value(scope)]);
        let first = self.variants.first().map_or(0, |variant| variant.discriminant);
        let discriminants = self.variants.iter().map(|variant| {
            let name = variant.ident.to_string();
            let disc = variant.discriminant;
            quote![#name => #disc]
        });
        let construction = self.variants.iter().enumerate().map(|(v, variant)| {
            let method = format_ident!("__expand_runtime_{}", variant.ident);
            let args = &variant.field_names;
            let types = variant.fields.iter().map(|field| &field.ty);
            let values = slots.iter().map(|(owner, index, field, ty)| {
                if *owner == v {
                    let arg = &args[*index];
                    quote![#field: #into_mut::expand_reference_value(#arg, scope)]
                } else {
                    quote![#field: <#ty as #ruda_type>::__expand_inactive_value(scope)]
                }
            });
            let disc = variant.discriminant;
            quote! {
                pub fn #method(scope: &mut #scope, #(#args: <#types as #ruda_type>::ExpandType),*) -> Self {
                    Self { discriminant: #disc.into(), value: #payload #turbofish { #(#values,)* } }
                }
            }
        });
        let constructors = self.with_constructors.then(|| {
            let constructors = self.variants.iter().map(|variant| {
                let base = format_ident!("new_{}", variant.ident);
                let expanded = format_ident!("__expand_new_{}", variant.ident);
                let construct = format_ident!("__expand_runtime_{}", variant.ident);
                let args = &variant.field_names;
                let types: Vec<_> = variant.fields.iter().map(|field| &field.ty).collect();
                quote! {
                    pub fn #base(#(#args: #types),*) -> Self { #core::unexpanded!() }
                    pub fn #expanded(scope: &mut #scope, #(#args: <#types as #ruda_type>::ExpandType),*) -> #expand #names {
                        #expand #turbofish::#construct(scope, #(#args),*)
                    }
                }
            });
            quote! {
                #[allow(non_snake_case, unused)]
                impl #generics #name #names #clause { #(#constructors)* }
            }
        });
        let accessors = self.runtime_field_accessors(true);
        quote! {
            #vis struct #payload #generics #clause { #(#fields,)* }
            impl #generics Clone for #payload #names #clause {
                fn clone(&self) -> Self { Self { #(#clones,)* } }
            }
            impl #generics #into_mut for #payload #names #clause {
                fn into_mut(self, scope: &mut #scope) -> Self { Self { #(#mutable,)* } }
                fn expand_reference_value(self, scope: &mut #scope) -> Self { Self { #(#snapshots,)* } }
            }
            impl #generics #debug for #payload #names #clause {}

            #vis struct #expand #generics #clause {
                discriminant: #native<i32>,
                value: #payload #names,
            }
            impl #generics Clone for #expand #names #clause {
                fn clone(&self) -> Self {
                    Self { discriminant: self.discriminant.clone(), value: self.value.clone() }
                }
            }
            impl #generics #into_mut for #expand #names #clause {
                fn into_mut(self, scope: &mut #scope) -> Self {
                    Self {
                        discriminant: #into_mut::into_mut(self.discriminant, scope),
                        value: #into_mut::into_mut(self.value, scope),
                    }
                }
                fn expand_reference_value(self, scope: &mut #scope) -> Self {
                    Self {
                        discriminant: #into_mut::expand_reference_value(self.discriminant, scope),
                        value: #into_mut::expand_reference_value(self.value, scope),
                    }
                }
            }
            impl #generics #debug for #expand #names #clause {}
            impl #generics #debug for #name #names #clause {}
            impl #generics #ruda_type for #name #names #clause {
                type ExpandType = #expand #names;
                fn __expand_inactive_value(scope: &mut #scope) -> Self::ExpandType {
                    #expand #turbofish { discriminant: #first.into(), value: #payload #turbofish { #(#inactive,)* } }
                }
            }
            impl #generics #ruda_enum for #expand #names #clause {
                type RuntimeValue = #payload #names;
                fn snapshot_for_match(self, scope: &mut #scope) -> Self {
                    #into_mut::expand_reference_value(self, scope)
                }
                fn discriminant(&self) -> #native<i32> { self.discriminant.clone() }
                fn runtime_value(self) -> Self::RuntimeValue { self.value }
                fn discriminant_of(variant_name: &'static str) -> i32 {
                    match variant_name { #(#discriminants,)* _ => unreachable!() }
                }
            }
            #[allow(non_snake_case)]
            impl #generics #expand #names #clause { #(#construction)* }
            #accessors
            #constructors
        }
    }

    fn generate_runtime_slots_launch(&self) -> TokenStream {
        let args_ty = self.args_ty();
        let launch_ty = self.args_ty_runtime();
        let compilation_ty = self.compilation_arg_ty();
        let launch_arg = prelude_type("LaunchArg");
        let launch_marker = prelude_type("LaunchMarker");
        let launcher = prelude_type("KernelLauncher");
        let builder = prelude_type("KernelBuilder");
        let ruda_type = prelude_type("RudaType");
        let name = &self.ident;
        let expand = &self.name_expand;
        let args = format_ident!("{}Args", name);
        let launch = format_ident!("{}Launch", name);
        let compilation = format_ident!("{}CompilationArg", name);
        let record = format_ident!("{}RuntimeCompilationArg", name);
        let vis = &self.vis;
        let (generics, names, _) = self.generics.split_for_impl();
        let turbofish = names.as_turbofish();
        let clause = self.launch_arg_where();
        let all = self.expanded_generics();
        let (_, all_names, _) = all.split_for_impl();
        let disc_arms = self.variants.iter().map(|variant| {
            let pattern = variant.runtime_pattern(&args);
            let disc = variant.discriminant;
            quote![#pattern => #disc]
        });
        let register_arms = self.variants.iter().map(|variant| {
            let pattern = variant.runtime_pattern(&args);
            let values = variant.field_names.iter().zip(&variant.fields).map(|(arg, field)| {
                let ty = &field.ty;
                quote![<#ty as #launch_arg>::register(#arg, launcher)]
            }).collect();
            let value = variant.runtime_record(&compilation, values);
            quote![#pattern => #value]
        });
        let expansion = |output: bool| {
            let method = if output { format_ident!("expand_output") } else { format_ident!("expand") };
            let arms = self.variants.iter().map(|variant| {
                let pattern = variant.runtime_pattern(&compilation);
                let locals = variant.field_names.iter().zip(&variant.fields).map(|(arg, field)| {
                    let ty = &field.ty;
                    quote![let #arg = <#ty as #launch_arg>::#method(#arg, builder);]
                });
                let args = &variant.field_names;
                let construct = format_ident!("__expand_runtime_{}", variant.ident);
                quote! {
                    #pattern => {
                        #(#locals)*
                        #expand #turbofish::#construct(&mut builder.scope, #(#args),*)
                    }
                }
            });
            quote! {
                let discriminant = arg.discriminant.as_ref()
                    .map(|discriminant| <i32 as #launch_arg>::expand(discriminant, builder));
                let mut value = match &arg.value { #(#arms,)* };
                if let ::core::option::Option::Some(discriminant) = discriminant {
                    value.discriminant = discriminant;
                }
                value
            }
        };
        let input = expansion(false);
        let output = expansion(true);
        quote! {
            #args_ty
            #launch_ty
            #compilation_ty
            #vis struct #record #generics #clause {
                discriminant: ::core::option::Option<<i32 as #launch_arg>::CompilationArg>,
                value: #compilation #names,
            }
            impl #generics Clone for #record #names #clause {
                fn clone(&self) -> Self { Self { discriminant: self.discriminant.clone(), value: self.value.clone() } }
            }
            impl #generics PartialEq for #record #names #clause {
                fn eq(&self, other: &Self) -> bool { self.discriminant == other.discriminant && self.value == other.value }
            }
            impl #generics Eq for #record #names #clause {}
            impl #generics core::hash::Hash for #record #names #clause {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    core::hash::Hash::hash(&self.discriminant, state);
                    core::hash::Hash::hash(&self.value, state);
                }
            }
            impl #generics core::fmt::Debug for #record #names #clause {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct(stringify!(#record)).field("discriminant", &self.discriminant).field("value", &self.value).finish()
                }
            }
            impl #generics #launch_arg for #name #names #clause {
                type RuntimeArg<R: #launch_marker> = #launch #all_names;
                type CompilationArg = #record #names;
                fn register<R: #launch_marker>(runtime_arg: Self::RuntimeArg<R>, launcher: &mut #launcher<R>) -> Self::CompilationArg {
                    let (value, runtime) = match runtime_arg {
                        #launch::Comptime(value) => (value, false),
                        #launch::Runtime(value) => (value, true),
                    };
                    let discriminant: i32 = match &value { #(#disc_arms,)* };
                    let discriminant = runtime.then(|| <i32 as #launch_arg>::register(discriminant, launcher));
                    let value = match value { #(#register_arms,)* };
                    #record #turbofish { discriminant, value }
                }
                fn expand(arg: &Self::CompilationArg, builder: &mut #builder) -> <Self as #ruda_type>::ExpandType { #input }
                fn expand_output(arg: &Self::CompilationArg, builder: &mut #builder) -> <Self as #ruda_type>::ExpandType { #output }
            }
        }
    }
}

impl RudaTypeVariant {
    fn runtime_pattern(&self, name: &Ident) -> TokenStream {
        let variant = &self.ident;
        let args = &self.field_names;
        match &self.fields {
            syn::Fields::Named(_) => quote![#name::#variant { #(#args,)* }],
            syn::Fields::Unnamed(_) => quote![#name::#variant(#(#args,)*)],
            syn::Fields::Unit => quote![#name::#variant],
        }
    }

    fn runtime_record(&self, name: &Ident, values: Vec<TokenStream>) -> TokenStream {
        let variant = &self.ident;
        let args = &self.field_names;
        match &self.fields {
            syn::Fields::Named(_) => quote![#name::#variant { #(#args: #values,)* }],
            syn::Fields::Unnamed(_) => quote![#name::#variant(#(#values,)*)],
            syn::Fields::Unit => quote![#name::#variant],
        }
    }
}
