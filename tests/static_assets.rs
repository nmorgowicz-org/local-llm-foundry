use llama_monitor::web::{gen_routes::static_routes, static_assets};

#[tokio::test]
async fn binary_brand_assets_are_served_as_bytes_with_mime() {
    let routes = static_routes();
    let response = warp::test::request()
        .path("/brand/token-ingot-192.png")
        .reply(&routes)
        .await;

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "image/png");
    assert_eq!(response.body(), static_assets::TOKEN_INGOT_192_PNG);
    assert!(response.body().starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[tokio::test]
async fn bundled_fonts_are_served_as_bytes_with_font_mime() {
    let routes = static_routes();
    let response = warp::test::request()
        .path("/fonts/inter/Inter-Regular.woff2")
        .reply(&routes)
        .await;

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "font/woff2");
    assert_eq!(response.body(), static_assets::INTER_REGULAR_WOFF2);
    assert!(!response.body().is_empty());
}

#[tokio::test]
async fn manifest_and_maskable_brand_assets_are_registered() {
    let routes = static_routes();
    let manifest = warp::test::request()
        .path("/manifest.json")
        .reply(&routes)
        .await;
    assert_eq!(manifest.status(), 200);
    let manifest_text = std::str::from_utf8(manifest.body()).expect("manifest is UTF-8");
    assert!(manifest_text.contains("token-ingot-maskable-512.png"));

    let maskable = warp::test::request()
        .path("/brand/token-ingot-maskable-512.png")
        .reply(&routes)
        .await;
    assert_eq!(maskable.status(), 200);
    assert_eq!(maskable.headers()["content-type"], "image/png");
    assert_eq!(maskable.body(), static_assets::TOKEN_INGOT_MASKABLE_512_PNG);
}

#[test]
fn production_svg_has_no_raster_or_executable_content() {
    let svg = static_assets::ICON_SVG;
    let lower = svg.to_ascii_lowercase();
    for forbidden in [
        "<script",
        "<image",
        "<filter",
        "foreignobject",
        "href=\"http",
    ] {
        assert!(
            !lower.contains(forbidden),
            "forbidden SVG content: {forbidden}"
        );
    }
    assert!(svg.contains("Token Ingot"));
}

#[test]
fn tray_template_preserves_layer_and_ingot_cutouts() {
    let decoder = png::Decoder::new(std::io::Cursor::new(
        static_assets::TOKEN_INGOT_TRAY_TEMPLATE_44_PNG,
    ));
    let mut reader = decoder.read_info().expect("tray template is a PNG");
    let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
    let output = reader.next_frame(&mut buffer).unwrap();
    assert_eq!((output.width, output.height), (44, 44));
    assert_eq!(output.color_type, png::ColorType::Rgba);
    let alpha = |x: usize, y: usize| buffer[(y * 44 + x) * 4 + 3];
    // These holes, not RGB shading, are what survives macOS template tinting.
    for (x, y) in [(0, 0), (22, 18), (22, 25), (18, 28), (22, 38)] {
        assert_eq!(alpha(x, y), 0, "missing template cutout at {x},{y}");
    }
    for (x, y) in [(22, 8), (22, 28), (10, 33)] {
        assert_eq!(alpha(x, y), 255, "missing solid face at {x},{y}");
    }
}

#[tokio::test]
async fn tray_template_is_registered_as_png_bytes() {
    let response = warp::test::request()
        .path("/brand/token-ingot-tray-template-44.png")
        .reply(&static_routes())
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "image/png");
    assert_eq!(
        response.body(),
        static_assets::TOKEN_INGOT_TRAY_TEMPLATE_44_PNG
    );
}
