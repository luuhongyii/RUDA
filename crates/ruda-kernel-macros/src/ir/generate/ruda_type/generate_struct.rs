use proc_macro2::TokenStream;
use darling::ast::Style;
use quote::{format_ident, quote};
use syn::{Ident, Index, Member, WhereClause};

use crate::ir::{
    generate::bounded_where_clause,
    parse::ruda_type::{RudaTypeStruct, TypeField},
    paths::prelude_type,
};

impl RudaTypeStruct {
    pub fn generate(&self, with_launch: bool) -> TokenStream {
        if with_launch {
            let launch_ty = self.launch_ty();
            let launch_new = self.launch_new();
            let launch_arg_impl = self.launch_arg_impl();

            quote! {
                #launch_ty
                #launch_new
                #launch_arg_impl
            }
        } else {
            let expand_ty = self.expand_ty();
            let clone_expand = self.clone_expand();
            let ruda_type_impl = self.ruda_type_impl();
            let expand_type_impl = self.expand_type_impl();
            let constructor = self.tuple_constructor();

            quote! {
                #expand_ty
                #clone_expand
                #ruda_type_impl
                #expand_type_impl
                #constructor
            }
        }
    }

    fn expand_ty(&self) -> proc_macro2::TokenStream {
        let fields = self.fields.iter().map(TypeField::expand_field);
        let name = &self.name_expand;
        let generics = &self.generics;
        let vis = &self.vis;
        let where_clause = &self.generics.where_clause;

        match self.style() {
            Style::Struct => quote![#vis struct #name #generics #where_clause { #(#fields),* }],
            Style::Tuple => quote![#vis struct #name #generics (#(#fields,)*) #where_clause;],
            Style::Unit => quote![#vis struct #name #generics #where_clause;],
        }
    }

    fn construct(&self, name: TokenStream, values: impl Iterator<Item = TokenStream>) -> TokenStream {
        match self.style() {
            Style::Struct => {
                let fields = self.fields.iter().zip(values).map(|(field, value)| {
                    let member = &field.ident;
                    quote![#member: #value]
                });
                quote![#name { #(#fields),* }]
            }
            Style::Tuple => quote![#name(#(#values,)*)],
            Style::Unit => quote![#name],
        }
    }

    fn tuple_constructor(&self) -> TokenStream {
        if self.style() != Style::Tuple { return quote![]; }
        let name = &self.ident;
        let name_expand = &self.name_expand;
        let scope = prelude_type("Scope");
        let ruda_type = prelude_type("RudaType");
        let into_mut = prelude_type("IntoMut");
        let vis = &self.vis;
        let (generics, names, where_clause) = self.generics.split_for_impl();
        let args = self.fields.iter().enumerate().map(|(i, field)| {
            let name = field.argument_ident(i);
            let ty = &field.ty;
            if field.comptime.is_present() { quote![#name: #ty] }
            else { quote![#name: <#ty as #ruda_type>::ExpandType] }
        });
        let values = self.fields.iter().enumerate().map(|(i, field)| {
            let name = field.argument_ident(i);
            if field.comptime.is_present() { quote![#name] }
            else { quote![#into_mut::expand_reference_value(#name, scope)] }
        });
        let accessors = self.fields.iter().enumerate().map(|(i, field)| {
            let accessor = format_ident!("__expand_tuple_field_reverse_{}", self.fields.len() - i - 1);
            let member = field.member(i);
            let ty = &field.ty;
            let ty = if field.comptime.is_present() { quote![#ty] }
                else { quote![<#ty as #ruda_type>::ExpandType] };
            quote! {
                pub fn #accessor(&self) -> #ty {
                    self.#member.clone()
                }
            }
        });
        quote! {
            impl #generics #name #names #where_clause {
                #vis fn expand(scope: &mut #scope, #(#args),*) -> #name_expand #names {
                    #name_expand(#(#values,)*)
                }
            }
            impl #generics #name_expand #names #where_clause {
                #(#accessors)*
            }
        }
    }

    fn clone_expand(&self) -> proc_macro2::TokenStream {
        let scope = prelude_type("Scope");
        let into_mut = prelude_type("IntoMut");

        let body = self.construct(quote![Self], self.fields.iter().enumerate().map(|(i, field)| {
            let member = field.member(i);
            quote![self.#member.clone()]
        }));
        let name = &self.name_expand;
        let generics = &self.generics;

        let (generics_impl, generics_use, where_clause) = generics.split_for_impl();

        quote! {
            impl #generics_impl Clone for #name #generics_use #where_clause {
                fn clone(&self) -> Self {
                    #body
                }
            }

            impl #generics_impl #name #generics_use #where_clause {
                pub fn __expand_clone_method(&self, scope: &mut #scope) -> Self {
                    #into_mut::expand_reference_value(self.clone(), scope)
                }
            }
        }
    }

    fn launch_ty(&self) -> proc_macro2::TokenStream {
        let name = &self.name_launch;
        let fields = self.fields.iter().enumerate().map(|(i, field)| field.launch_field(i));
        let generics = self.expanded_generics();
        let where_clause = self.launch_arg_where();
        let vis = &self.vis;

        quote! {
            #vis struct #name #generics #where_clause {
                _phantom_runtime: core::marker::PhantomData<R>,
                #(#fields),*
            }
        }
    }

    fn launch_new(&self) -> proc_macro2::TokenStream {
        let args = self.fields.iter().enumerate().map(|(i, field)| field.launch_new_arg(i));
        let fields = self.fields.iter().enumerate().map(|(i, field)| field.argument_ident(i));
        let name = &self.name_launch;

        let generics = self.expanded_generics();
        let (generics_impl, generics_use, _) = generics.split_for_impl();
        let where_clause = self.launch_arg_where();
        let vis = &self.vis;

        quote! {
            impl #generics_impl #name #generics_use #where_clause {
                /// New kernel
                #[allow(clippy::too_many_arguments)]
                #vis fn new(#(#args),*) -> Self {
                    Self {
                        _phantom_runtime: core::marker::PhantomData,
                        #(#fields),*
                    }
                }
            }
        }
    }

    fn register_impl(&self) -> proc_macro2::TokenStream {
        let kernel_launcher = prelude_type("KernelLauncher");
        let launch_marker = prelude_type("LaunchMarker");
        let launch_arg = prelude_type("LaunchArg");
        let compilation_ty = self.compilation_ty_ident();
        let fields = self.fields.iter().enumerate().map(|(i, field)| {
            let ident = field.argument_ident(i);
            let ty = &field.ty;
            if field.comptime.is_present() { quote![#ident: arg.#ident] }
            else { quote![#ident: <#ty as #launch_arg>::register(arg.#ident, launcher)] }
        });

        quote! {
            fn register<R: #launch_marker>(arg: Self::RuntimeArg<R>, launcher: &mut #kernel_launcher<R>) -> Self::CompilationArg {
                #compilation_ty {
                    #(#fields),*
                }
            }
        }
    }

    fn ruda_type_impl(&self) -> proc_macro2::TokenStream {
        let ruda_type = prelude_type("RudaType");
        let scope = prelude_type("Scope");
        let name = &self.ident;
        let name_expand = &self.name_expand;

        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let inactive = if self.fields.iter().any(|field| field.comptime.is_present()) {
            quote![]
        } else {
            let turbofish = generic_names.as_turbofish();
            let value = self.construct(quote![#name_expand #turbofish], self.fields.iter().map(|field| {
                let ty = &field.ty;
                quote![<#ty as #ruda_type>::__expand_inactive_value(scope)]
            }));
            quote! {
                fn __expand_inactive_value(scope: &mut #scope) -> Self::ExpandType {
                    #value
                }
            }
        };

        quote! {
            impl #generics #ruda_type for #name #generic_names #where_clause {
                type ExpandType = #name_expand #generic_names;
                #inactive
            }
        }
    }

    fn compilation_ty_ident(&self) -> Ident {
        Ident::new(
            format!("{}CompilationArg", self.ident).as_str(),
            self.ident.span(),
        )
    }

    fn compilation_ty(&self, name: &Ident) -> proc_macro2::TokenStream {
        let name_debug = &self.ident;
        let fields = self.fields.iter().enumerate().map(|(i, field)| field.compilation_arg_field(i));
        let generics = &self.generics;
        let (type_generics_names, impl_generics, _) = self.generics.split_for_impl();
        let where_generics = self.launch_arg_where();
        let vis = &self.vis;

        fn generate<'a, F: Fn(&Ident) -> TokenStream>(
            fields: impl Iterator<Item = &'a TypeField>,
            func: F,
        ) -> Vec<TokenStream> {
            fields
                .enumerate()
                .map(|(i, field)| func(&field.argument_ident(i)))
                .collect::<Vec<_>>()
        }

        let clone = generate(self.fields.iter(), |name| quote!(#name: self.#name.clone()));
        let hash = generate(self.fields.iter(), |name| quote!(self.#name.hash(state)));
        let partial_eq = generate(
            self.fields.iter(),
            |name| quote!(self.#name.eq(&other.#name)),
        );
        let debug = generate(
            self.fields.iter(),
            |name| quote!(.field(stringify!(#name), &self.#name)),
        );

        quote! {
            #vis struct #name #generics #where_generics {
                #(#fields),*
            }

            impl #type_generics_names Clone for #name #impl_generics #where_generics {
                fn clone(&self) -> Self {
                    Self {
                        #(#clone,)*
                    }
                }
            }

            impl #type_generics_names core::hash::Hash for #name #impl_generics #where_generics {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    #(#hash;)*
                }
            }

            impl #type_generics_names core::cmp::PartialEq for #name #impl_generics #where_generics {
                fn eq(&self, other: &Self) -> bool {
                    #(#partial_eq &&)* true
                }
            }

            impl #type_generics_names core::fmt::Debug for #name #impl_generics #where_generics {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct(stringify!(#name_debug))
                    #(#debug)*
                    .finish()
                }
            }
            impl #type_generics_names core::cmp::Eq for #name #impl_generics #where_generics { }
        }
    }

    fn launch_arg_impl(&self) -> proc_macro2::TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = &self.ident;
        let name_launch = &self.name_launch;
        let name_expand = &self.name_expand;
        let body_input = self.construct(quote![#name_expand], self.fields.iter().enumerate().map(|(i, field)| {
            let name = field.argument_ident(i);
            let ty = &field.ty;
            if field.comptime.is_present() { quote![arg.#name.clone()] }
            else { quote![<#ty as #launch_arg>::expand(&arg.#name, builder)] }
        }));
        let body_output = self.construct(quote![#name_expand], self.fields.iter().enumerate().map(|(i, field)| {
            let name = field.argument_ident(i);
            let ty = &field.ty;
            if field.comptime.is_present() { quote![arg.#name.clone()] }
            else { quote![<#ty as #launch_arg>::expand_output(&arg.#name, builder)] }
        }));

        let (type_generics, type_generic_names, _) = self.generics.split_for_impl();
        let where_clause = self.launch_arg_where();

        let register_impl = self.register_impl();

        let (_, compilation_generics, _) = self.generics.split_for_impl();
        let assoc_generics = self.assoc_generics();
        let all = self.expanded_generics();
        let (_, all_generic_names, _) = all.split_for_impl();

        let compilation_ident = self.compilation_ty_ident();
        let compilation_arg = self.compilation_ty(&compilation_ident);

        quote! {
            #compilation_arg

            impl #type_generics #launch_arg for #name #type_generic_names #where_clause {
                type RuntimeArg #assoc_generics = #name_launch #all_generic_names;
                type CompilationArg = #compilation_ident #compilation_generics;

                #register_impl

                fn expand(
                    arg: &Self::CompilationArg,
                    builder: &mut KernelBuilder,
                ) -> <Self as RudaType>::ExpandType {
                    #body_input
                }
                fn expand_output(
                    arg: &Self::CompilationArg,
                    builder: &mut KernelBuilder,
                ) -> <Self as RudaType>::ExpandType {
                    #body_output
                }
            }
        }
    }

    fn expand_type_impl(&self) -> proc_macro2::TokenStream {
        let into_mut = prelude_type("IntoMut");
        let debug = prelude_type("RudaDebug");
        let scope = prelude_type("Scope");
        let name_expand = &self.name_expand;
        let (generics, generic_names, where_clause) = self.generics.split_for_impl();
        let body = self.construct(quote![Self], self.fields.iter().enumerate().map(|(i, field)| {
            let member = field.member(i);
            if field.comptime.is_present() { quote![self.#member] }
            else { quote![#into_mut::into_mut(self.#member, scope)] }
        }));
        let snapshot = self.construct(quote![Self], self.fields.iter().enumerate().map(|(i, field)| {
            let member = field.member(i);
            if field.comptime.is_present() { quote![self.#member] }
            else { quote![#into_mut::expand_reference_value(self.#member, scope)] }
        }));

        quote! {
            impl #generics #into_mut for #name_expand #generic_names #where_clause {
                fn into_mut(self, scope: &mut #scope) -> Self {
                    #body
                }
                fn expand_reference_value(self, scope: &mut #scope) -> Self {
                    #snapshot
                }
            }

            impl #generics #debug for #name_expand #generic_names #where_clause {}
        }
    }

    fn launch_arg_where(&self) -> Option<WhereClause> {
        if self.skip_bounds.is_present() {
            return self.generics.where_clause.clone();
        }
        let launch_arg = prelude_type("LaunchArg");
        let fields = self
            .fields
            .iter()
            .filter(|it| !it.comptime.is_present())
            .cloned();
        let mut generics = self.generics.clone();
        generics.where_clause = bounded_where_clause(&self.generics, fields, |param| quote![#param: #launch_arg]);
        let compilation_arg = prelude_type("CompilationArg");
        let comptime_fields = self.fields.iter().filter(|it| it.comptime.is_present()).cloned();
        bounded_where_clause(&generics, comptime_fields, |param| quote![#param: #compilation_arg])
    }
}

impl TypeField {
    fn argument_ident(&self, index: usize) -> Ident {
        self.ident.clone().unwrap_or_else(|| format_ident!("arg_{index}"))
    }

    fn member(&self, index: usize) -> Member {
        self.ident.clone().map(Member::Named).unwrap_or_else(|| Member::Unnamed(Index::from(index)))
    }

    pub fn expand_field(&self) -> TokenStream {
        let ruda_type = prelude_type("RudaType");
        let vis = &self.vis;
        let name = self.ident.as_ref().map(|name| quote![#name:]);
        let ty = &self.ty;
        if self.comptime.is_present() {
            quote![#vis #name #ty]
        } else {
            quote![#vis #name <#ty as #ruda_type>::ExpandType]
        }
    }

    pub fn launch_field(&self, index: usize) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.argument_ident(index);
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::RuntimeArg<R>]
        } else {
            quote![#vis #name: #ty]
        }
    }

    pub fn launch_new_arg(&self, index: usize) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let name = self.argument_ident(index);
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#name: <#ty as #launch_arg>::RuntimeArg<R>]
        } else {
            quote![#name: #ty]
        }
    }

    pub fn compilation_arg_field(&self, index: usize) -> TokenStream {
        let launch_arg = prelude_type("LaunchArg");
        let vis = &self.vis;
        let name = self.argument_ident(index);
        let ty = &self.ty;

        if !self.comptime.is_present() {
            quote![#vis #name: <#ty as #launch_arg>::CompilationArg]
        } else {
            quote![#vis #name: #ty]
        }
    }

}
