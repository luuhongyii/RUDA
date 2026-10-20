use ident_case::RenameRule;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{Ident, parse_quote};

use crate::ir::{
    parse::kernel::{AddressType, GenericArg, KernelParam, Launch},
    paths::{core_type, prelude_type},
};

impl ToTokens for Launch {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let vis = &self.vis;

        let name = &self.func.sig.name;
        let launch = self.launch();
        let prepare = self.prepare();
        let launch_unchecked = self.launch_unchecked();
        let launch_erased = self.erased_launch(false, false);
        let launch_unchecked_erased = self.erased_launch(true, false);
        let prepare_erased = self.erased_launch(false, true);
        let aliases = self.create_type_alias();
        let dummy = self.create_dummy_kernel();
        let dummy_erased = self.create_dummy_kernel_for(true);
        let kernel = self.kernel_definition();
        let mut func = self.func.clone();
        func.sig.name = format_ident!("expand");
        let func = func.to_tokens_mut();

        let out = quote! {
            #vis mod #name {
                use super::*;

                #aliases

                #[allow(unused, clippy::all)]
                #func

                #kernel
                #launch
                #launch_unchecked
                #prepare
                #launch_erased
                #launch_unchecked_erased
                #prepare_erased
                #dummy
                #dummy_erased
            }
        };

        if self.args.debug.is_present() {
            let file = syn::parse_file(&out.to_string()).unwrap();
            let tokens = prettyplease::unparse(&file);
            panic!("{tokens}");
        }
        tokens.extend(out);
    }
}

impl Launch {
    fn launch(&self) -> TokenStream {
        if self.args.launch.is_present() {
            let compute_client = prelude_type("ComputeClient");
            let ruda_count = prelude_type("RudaCount");
            let ruda_dim = prelude_type("RudaDim");
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = &self.launch_generics;
            let where_clause = &generics.where_clause;
            let args = self.launch_args();
            let body = self.launch_body();

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn launch #generics(
                    __client: &#compute_client<__R>,
                    __ruda_count: #ruda_count,
                    __ruda_dim: #ruda_dim,
                    #address_type
                    #(#args),*
                ) #where_clause {
                    #body
                    launcher.launch(__ruda_count, __kernel, __client)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn launch_unchecked(&self) -> TokenStream {
        if self.args.launch_unchecked.is_present() {
            let compute_client = prelude_type("ComputeClient");
            let ruda_count = prelude_type("RudaCount");
            let ruda_dim = prelude_type("RudaDim");
            let address_type = prelude_type("AddressType");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime without bound checks.\n\n\
                 # Safety\n\n\
                 The kernel must not:\n\
                 - Contain any out of bounds reads or writes. Doing so is immediate UB.\n\
                 - Contain any loops that never terminate. These may be optimized away entirely or cause\n\
                   other unpredictable behaviour.",
                self.func.sig.name
            );
            let generics = &self.launch_generics;
            let where_clause = &generics.where_clause;
            let args = self.launch_args();
            let body = self.launch_body();

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub unsafe fn launch_unchecked #generics(
                    __client: &#compute_client<__R>,
                    __ruda_count: #ruda_count,
                    __ruda_dim: #ruda_dim,
                    #address_type
                    #(#args),*
                ) #where_clause {
                    #body
                    launcher.launch_unchecked(__ruda_count, __kernel, __client)
                }
            }
        } else {
            TokenStream::new()
        }
    }

    fn prepare(&self) -> TokenStream {
        if !self.args.launch.is_present() && !self.args.launch_unchecked.is_present() {
            return TokenStream::new();
        }
        let compute_client = prelude_type("ComputeClient");
        let prepared = prelude_type("PreparedKernel");
        let ruda_count = prelude_type("RudaCount");
        let ruda_dim = prelude_type("RudaDim");
        let address_type = prelude_type("AddressType");
        let generics = &self.launch_generics;
        let where_clause = &generics.where_clause;
        let args = self.launch_args();
        let body = self.launch_body();
        let address_type = match self.args.address_type {
            AddressType::Dynamic => quote![__address_type: #address_type,],
            _ => quote![],
        };
        quote! {
            /// Register owned arguments for explicit graph construction.
            /// No kernel is compiled or executed by this function.
            #[allow(clippy::too_many_arguments)]
            pub fn prepare #generics(
                __client: &#compute_client<__R>,
                __ruda_count: #ruda_count,
                __ruda_dim: #ruda_dim,
                #address_type
                #(#args),*
            ) -> #prepared<__R> #where_clause {
                #body
                launcher.prepare(__ruda_count, __kernel, __client)
            }
        }
    }

    fn launch_body(&self) -> TokenStream {
        self.launch_body_for(quote![__R])
    }

    fn erased_generics(&self) -> syn::Generics {
        let mut generics = self.launch_generics.clone();
        generics.params = generics.params.into_iter().filter(|param| {
            !matches!(param, syn::GenericParam::Type(ty) if ty.ident == "__R")
        }).collect();
        generics
    }

    fn kernel_type_args(&self, marker: &TokenStream) -> TokenStream {
        let args = self.kernel_generics.params.iter().map(|param| match param {
            syn::GenericParam::Type(ty) if ty.ident == "__R" => marker.clone(),
            syn::GenericParam::Type(ty) => {
                let name = &ty.ident;
                quote![#name]
            }
            syn::GenericParam::Const(param) => {
                let name = &param.ident;
                quote![#name]
            }
            syn::GenericParam::Lifetime(param) => {
                let lifetime = &param.lifetime;
                quote![#lifetime]
            }
        });
        quote![::<#(#args),*>]
    }

    fn erased_launch(&self, unchecked: bool, prepare: bool) -> TokenStream {
        let enabled = if prepare {
            self.args.launch.is_present() || self.args.launch_unchecked.is_present()
        } else if unchecked {
            self.args.launch_unchecked.is_present()
        } else {
            self.args.launch.is_present()
        };
        if !enabled { return TokenStream::new(); }
        let client = prelude_type("Client");
        let count = prelude_type("RudaCount");
        let dim = prelude_type("RudaDim");
        let address_type = prelude_type("AddressType");
        let generics = self.erased_generics();
        let where_clause = &generics.where_clause;
        let args = self.launch_args_for(parse_quote![()]);
        let body = self.launch_body_for(quote![()]);
        let address_type = match self.args.address_type {
            AddressType::Dynamic => quote![__address_type: #address_type,],
            _ => quote![],
        };
        let (name, unsafety, output, submit, doc) = if prepare {
            let prepared = prelude_type("ErasedPreparedKernel");
            (format_ident!("prepare_erased"), quote![], quote![-> #prepared],
                quote![launcher.prepare_erased(__ruda_count, __kernel, __client)],
                "Prepare an owned invocation on a fixed client queue without selecting a Rust runtime type.")
        } else if unchecked {
            (format_ident!("launch_unchecked_erased"), quote![unsafe], quote![],
                quote![unsafe { launcher.launch_unchecked_erased(__ruda_count, __kernel, __client) }],
                "Launch on the selected client without a Rust runtime type.\n\n# Safety\nThe kernel must have valid memory accesses and terminate.")
        } else {
            (format_ident!("launch_erased"), quote![], quote![],
                quote![launcher.launch_erased(__ruda_count, __kernel, __client)],
                "Launch on the selected client without selecting a Rust runtime type.")
        };
        quote! {
            #[allow(clippy::too_many_arguments)]
            #[track_caller]
            #[doc = #doc]
            pub #unsafety fn #name #generics(
                __client: &#client,
                __ruda_count: #count,
                __ruda_dim: #dim,
                #address_type
                #(#args),*
            ) #output #where_clause {
                #body
                #submit
            }
        }
    }

    fn launch_body_for(&self, marker: TokenStream) -> TokenStream {
        let kernel_name = self.kernel_name();
        let kernel_generics = self.kernel_type_args(&marker);
        let comptime_args = self.comptime_params().map(|it| &it.name);
        let (setup, args) = self.launcher_setup(marker, quote![__client.properties()]);
        quote! {
            #setup
            let __kernel = #kernel_name #kernel_generics::new(__settings, __client.clone(), #args #(#comptime_args),*);
        }
    }

    fn launcher_setup(&self, marker: TokenStream, properties: TokenStream) -> (TokenStream, TokenStream) {
        let kernel_launcher = prelude_type("KernelLauncher");

        let mappings = self.func.sig.define_mappings();
        let generic_registers =
            self.func
                .analysis
                .register_types(mappings, quote![scope], false, true);

        let settings = self.configure_settings();
        let (registers, args) = self.arg_registers();

        let setup = quote! {
            #settings

            let mut launcher = #kernel_launcher::<#marker>::new(__settings.clone());
            launcher.with_scope(|scope| {
                scope.device_properties(#properties);
                #generic_registers
            });

            #registers
        };
        (setup, args)
    }

    fn configure_settings(&self) -> TokenStream {
        let kernel_settings = prelude_type("KernelSettings");
        let addr_ty = prelude_type("AddressType");
        let address_type = match self.args.address_type {
            AddressType::U32 => quote![#addr_ty::U32],
            AddressType::U64 => quote![#addr_ty::U64],
            AddressType::Dynamic => quote![__address_type],
        };

        quote! {
            let mut __settings = #kernel_settings::default()
                .ruda_dim(__ruda_dim).address_type(#address_type);
        }
    }

    fn create_type_alias(&self) -> TokenStream {
        let mut aliases = quote! {};
        if !self.func.args.explicit_define.is_present() {
            for (
                name,
                GenericArg {
                    expand_ty,
                    marker_ty,
                    ..
                },
            ) in self.func.analysis.map.iter()
            {
                aliases.extend(quote! {
                    pub struct #marker_ty;
                    /// Type to be used as a generic for launch kernel argument.
                    pub type #name = #expand_ty;
                });
            }
        }

        aliases
    }
    fn create_dummy_kernel(&self) -> TokenStream {
        self.create_dummy_kernel_for(false)
    }

    fn create_dummy_kernel_for(&self, erased: bool) -> TokenStream {
        if self.args.create_dummy_kernel.is_present() {
            let ruda_count = prelude_type("RudaCount");
            let ruda_dim = prelude_type("RudaDim");
            let address_type = prelude_type("AddressType");
            let private = core_type("__private");
            let ir = core_type("ir");
            let context = prelude_type("KernelContext");

            let kernel_doc = format!(
                "Launch the kernel [{}()] on the given runtime",
                self.func.sig.name
            );
            let generics = if erased { self.erased_generics() } else { self.launch_generics.clone() };
            let where_clause = &generics.where_clause;
            let marker = if erased { quote![()] } else { quote![__R] };
            let generic_names = self.kernel_type_args(&marker);
            let name = if erased { format_ident!("create_dummy_kernel_erased") } else { format_ident!("create_dummy_kernel") };

            let kernel_name = self.kernel_name();
            let comptime_args = self.launch_args_for(if erased { parse_quote![()] } else { parse_quote![__R] });
            let comptime_names = self.comptime_params().map(|it| &it.name);
            let (setup, args) = self.launcher_setup(marker, quote![&__device_properties]);

            let address_type = match self.args.address_type {
                AddressType::Dynamic => quote![__address_type: #address_type,],
                _ => quote![],
            };

            quote! {
                #[allow(clippy::too_many_arguments)]
                #[doc = #kernel_doc]
                pub fn #name #generics(
                    __device_properties: #private::Arc<#ir::DeviceProperties>,
                    __target_properties: #private::Arc<#ir::TargetProperties>,
                    __ruda_count: #ruda_count,
                    __ruda_dim: #ruda_dim,
                    #address_type
                    #(#comptime_args),*
                ) -> #kernel_name #generic_names #where_clause {
                    #setup

                    launcher.discard();
                    #kernel_name #generic_names::new(
                        __settings,
                        #context::from_properties(__device_properties, __target_properties),
                        #args #(#comptime_names),*
                    )
                }
            }
        } else {
            TokenStream::new()
        }
    }

    pub fn runtime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func.sig.runtime_params()
    }

    fn launch_args(&self) -> Vec<KernelParam> {
        self.launch_args_for(parse_quote![__R])
    }

    fn launch_args_for(&self, marker: syn::Type) -> Vec<KernelParam> {
        let mut args = self.func.sig.parameters.clone();
        let runtime_arg = core_type("RuntimeArg");
        for arg in args.iter_mut().filter(|it| !it.is_const) {
            let ty = arg.ty_owned();
            arg.normalized_ty = parse_quote![#runtime_arg<#ty, #marker>];
            arg.mut_token = None;
        }
        args
    }

    pub fn kernel_name(&self) -> Ident {
        let kernel_name = RenameRule::PascalCase.apply_to_field(self.func.sig.name.to_string());
        format_ident!("{kernel_name}")
    }

    pub fn comptime_params(&self) -> impl Iterator<Item = &KernelParam> {
        self.func
            .sig
            .parameters
            .iter()
            .filter(|param| param.is_const)
    }
}
