use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, parse_quote};

type Derive = fn(&DeriveInput) -> TokenStream;

fn compare_owners(input: DeriveInput, default: Derive, native: Derive) {
    let before = input.to_token_stream().to_string();
    let old = default(&input);
    let new = native(&input);
    syn::parse2::<syn::File>(old.clone()).expect("default generated syntax");
    syn::parse2::<syn::File>(new.clone()).expect("native generated syntax");
    let old = old.to_string();
    let new = new.to_string();
    assert!(old.contains(":: ruda_model ::"));
    assert!(new.contains(":: ruda_model ::"));
    assert!(!new.contains("ruda ::"));
    assert_eq!(old, new);
    assert_eq!(before, input.to_token_stream().to_string());
}

#[test]
fn config_struct_owners() {
    compare_owners(
        parse_quote! {
            pub struct Settings {
                value: usize,
                optional: Option<String>,
                #[config(default = 2)]
                count: usize,
            }
        },
        crate::derive_config,
        crate::derive_config_native,
    );
}

#[test]
fn config_enum_owners() {
    compare_owners(
        parse_quote! { pub enum Settings { Empty, Value(usize), Named { value: String } } },
        crate::derive_config,
        crate::derive_config_native,
    );
}

#[test]
fn module_struct_and_enum_owners() {
    for input in [
        parse_quote! {
            pub struct Layer<B: Backend> {
                weight: Param<Tensor<B, 2>>,
                #[module(skip)]
                label: String,
            }
        },
        parse_quote! { pub enum Layer<B: Backend> { One(Linear<B>), Two(Embedding<B>) } },
    ] {
        compare_owners(input, crate::derive_module, crate::derive_module_native);
    }
}

#[test]
fn constant_module_owners() {
    compare_owners(
        parse_quote! { pub struct Constant { value: usize } },
        crate::derive_module,
        crate::derive_module_native,
    );
}

#[test]
fn record_struct_enum_and_backendless_owners() {
    for input in [
        parse_quote! { pub struct State<B: Backend> { weight: Tensor<B, 2> } },
        parse_quote! { pub enum State<B: Backend> { One(Tensor<B, 2>), Two(Tensor<B, 1>) } },
        parse_quote! { pub struct State { count: usize } },
    ] {
        compare_owners(input, crate::derive_record, crate::derive_record_native);
    }
}

#[test]
fn config_user_paths_and_literals_are_not_rewritten() {
    let input: DeriveInput = parse_quote! {
        pub enum UserSettings {
            #[serde(rename = "ruda::serde")]
            Value(ruda::UserValue),
        }
    };
    let before = input.to_token_stream().to_string();
    for derive in [crate::derive_config as Derive, crate::derive_config_native] {
        let output = derive(&input).to_string();
        assert!(output.contains(&quote!(ruda::UserValue).to_string()));
        assert!(output.contains(&quote!(#[serde(rename = "ruda::serde")]).to_string()));
        assert_eq!(before, input.to_token_stream().to_string());
    }
}

#[test]
fn module_and_record_user_backend_paths_are_not_rewritten() {
    struct UserTypes {
        backend: syn::Path,
        field: syn::Type,
        found_backend: bool,
        found_field: bool,
    }
    impl<'ast> syn::visit::Visit<'ast> for UserTypes {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            self.found_backend |= *path == self.backend;
            syn::visit::visit_path(self, path);
        }
        fn visit_type(&mut self, ty: &'ast syn::Type) {
            self.found_field |= *ty == self.field;
            syn::visit::visit_type(self, ty);
        }
    }
    let input: DeriveInput = parse_quote! {
        pub struct UserModule<B: ruda::CustomBackend> {
            value: ruda::module::Param<ruda::tensor::Tensor<B, 2>>,
        }
    };
    let before = input.to_token_stream().to_string();
    for derive in [
        crate::derive_module as Derive,
        crate::derive_module_native,
        crate::derive_record,
        crate::derive_record_native,
    ] {
        let output = syn::parse2::<syn::File>(derive(&input)).expect("generated Rust syntax");
        let mut types = UserTypes {
            backend: parse_quote!(ruda::CustomBackend),
            field: parse_quote!(ruda::module::Param<ruda::tensor::Tensor<B, 2>>),
            found_backend: false,
            found_field: false,
        };
        syn::visit::Visit::visit_file(&mut types, &output);
        assert!(types.found_backend && types.found_field);
        assert_eq!(before, input.to_token_stream().to_string());
    }
}
