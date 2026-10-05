use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "webview-popover")]
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// Attempt a silent winget install of the WebView2 Evergreen runtime.
///
/// Guarded by a `Once` so the install is only attempted a single time per process
/// lifetime even if the popover is opened repeatedly. Failures are logged but never
/// propagate — this is best-effort. Only compiled on Windows when the WebView2
/// popover is built (its only caller is the popover's failure path).
#[cfg(all(windows, feature = "webview-popover"))]
fn try_install_webview2() {
    use std::sync::Once;
    static INSTALL_ONCE: Once = Once::new();
    INSTALL_ONCE.call_once(|| {
        eprintln!(
            "[tray] WebView2 runtime appears missing. \
             Attempting silent install via winget \
             (Microsoft.EdgeWebView2Runtime). \
             If this fails, download from: \
             https://developer.microsoft.com/microsoft-edge/webview2/"
        );
        let status = crate::platform::no_window(&mut std::process::Command::new("winget"))
            .args([
                "install",
                "-e",
                "--id",
                "Microsoft.EdgeWebView2Runtime",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--disable-interactivity",
            ])
            .status();
        match status {
            Ok(s) if s.success() => {
                eprintln!(
                    "[tray] winget install Microsoft.EdgeWebView2Runtime succeeded. \
                     Please restart Local LLM Foundry to enable the tray popover."
                );
            }
            Ok(s) => {
                eprintln!(
                    "[tray] winget install Microsoft.EdgeWebView2Runtime exited with \
                     status {s}. Install it manually from: \
                     https://developer.microsoft.com/microsoft-edge/webview2/"
                );
            }
            Err(e) => {
                eprintln!(
                    "[tray] winget not found or failed to launch ({e}). \
                     Install WebView2 runtime manually from: \
                     https://developer.microsoft.com/microsoft-edge/webview2/"
                );
            }
        }
    });
}

#[cfg(feature = "webview-popover")]
use tray_icon::TrayIconEvent;
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use winit::application::ApplicationHandler;
#[cfg(feature = "webview-popover")]
use winit::dpi::PhysicalPosition;
#[cfg(feature = "webview-popover")]
use winit::dpi::PhysicalSize;
#[cfg(all(not(target_os = "linux"), feature = "webview-popover"))]
use winit::dpi::Position;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;
#[cfg(all(not(target_os = "linux"), feature = "webview-popover"))]
use winit::window::{WindowAttributes, WindowLevel};

#[cfg(target_os = "macos")]
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};

#[cfg(all(target_os = "linux", feature = "webview-popover"))]
use gtk::prelude::*;
#[cfg(all(target_os = "linux", feature = "webview-popover"))]
use wry::WebViewBuilderExtUnix;

use crate::gpu::GpuMetrics;
use crate::llama::metrics::LlamaMetrics;
use crate::state::AppState;
use crate::system::SystemMetrics;

#[cfg(feature = "webview-popover")]
const POPOVER_WIDTH: f64 = 240.0;
#[cfg(feature = "webview-popover")]
const POPOVER_MIN_WIDTH: f64 = 200.0;
#[cfg(feature = "webview-popover")]
const POPOVER_MAX_WIDTH: f64 = 520.0;
#[cfg(feature = "webview-popover")]
const POPOVER_INITIAL_HEIGHT: f64 = 220.0;
#[cfg(feature = "webview-popover")]
const POPOVER_MIN_HEIGHT: f64 = POPOVER_INITIAL_HEIGHT;
#[cfg(feature = "webview-popover")]
const POPOVER_MAX_HEIGHT: f64 = 520.0;

type TrayMetrics = (
    SystemMetrics,
    Option<Vec<(String, GpuMetrics)>>,
    LlamaMetrics,
    Option<String>,
);

fn create_tray_icon() -> Icon {
    // macOS tints only alpha, so use separated layers rather than flattening
    // the overlapping color mark. tray-icon caps 44px at 22pt for Retina.
    #[cfg(target_os = "macos")]
    let png_bytes = crate::web::static_assets::TOKEN_INGOT_TRAY_TEMPLATE_44_PNG;
    #[cfg(not(target_os = "macos"))]
    let png_bytes = crate::web::static_assets::TOKEN_INGOT_22_PNG;
    let decoded = (|| {
        let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
        let mut reader = decoder.read_info().ok()?;
        let mut buffer = vec![0; reader.output_buffer_size()?];
        let output = reader.next_frame(&mut buffer).ok()?;
        let bytes = &buffer[..output.buffer_size()];
        let rgba = match output.color_type {
            png::ColorType::Rgba => bytes.to_vec(),
            png::ColorType::Rgb => bytes
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
                .collect(),
            png::ColorType::GrayscaleAlpha => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[0], pixel[0], pixel[1]])
                .collect(),
            png::ColorType::Grayscale => bytes
                .iter()
                .flat_map(|pixel| [*pixel, *pixel, *pixel, 255])
                .collect(),
            _ => return None,
        };
        Some((rgba, output.width, output.height))
    })();

    decoded
        .and_then(|(rgba, width, height)| Icon::from_rgba(rgba, width, height).ok())
        .unwrap_or_else(|| Icon::from_rgba(vec![0, 0, 0, 255], 1, 1).unwrap())
}

pub fn run_tray(state: AppState, port: u16, app_root: PathBuf) -> anyhow::Result<()> {
    #[cfg(not(feature = "webview-popover"))]
    let _ = port;

    #[cfg(target_os = "macos")]
    {
        let _ = mac_notification_sys::set_application("com.apple.Finder");
    }

    #[cfg(target_os = "macos")]
    let event_loop = EventLoop::builder()
        .with_activation_policy(ActivationPolicy::Accessory)
        .build()?;

    #[cfg(not(target_os = "macos"))]
    let event_loop = EventLoop::builder().build()?;

    #[cfg(all(target_os = "linux", feature = "webview-popover"))]
    let gtk_ready = match gtk::init() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[tray] GTK init failed; tray WebView popover disabled: {e}");
            false
        }
    };

    #[cfg(feature = "webview-popover")]
    let (popover_tx, popover_rx) = mpsc::channel();
    let tray_start_failed = Arc::new(AtomicBool::new(false));

    let app: Box<dyn winit::application::ApplicationHandler + 'static> = Box::new(TrayApp {
        tray_state: TrayState {
            app_state: Arc::new(state),
        },
        tray: None,
        icon: create_tray_icon(),
        port,
        app_root,
        menu_quit_id: None,
        menu_open_id: None,
        menu_open_logs_id: None,
        #[cfg(feature = "webview-popover")]
        popover: None,
        #[cfg(feature = "webview-popover")]
        popover_tx,
        #[cfg(feature = "webview-popover")]
        popover_rx,
        tray_start_failed: Arc::clone(&tray_start_failed),
        #[cfg(all(target_os = "linux", feature = "webview-popover"))]
        gtk_ready,
    });

    event_loop.run_app(app)?;

    if tray_start_failed.load(Ordering::Relaxed) {
        anyhow::bail!("failed to create tray icon");
    }

    Ok(())
}

struct TrayApp {
    tray_state: TrayState,
    tray: Option<TrayIcon>,
    icon: Icon,
    port: u16,
    app_root: PathBuf,
    /// Right-click context-menu item ids (so every platform/config has a way to quit
    /// and open the dashboard, independent of the WebView2 popover).
    menu_quit_id: Option<tray_icon::menu::MenuId>,
    menu_open_id: Option<tray_icon::menu::MenuId>,
    menu_open_logs_id: Option<tray_icon::menu::MenuId>,
    #[cfg(feature = "webview-popover")]
    popover: Option<Popover>,
    #[cfg(feature = "webview-popover")]
    popover_tx: Sender<PopoverMessage>,
    #[cfg(feature = "webview-popover")]
    popover_rx: Receiver<PopoverMessage>,
    tray_start_failed: Arc<AtomicBool>,
    #[cfg(all(target_os = "linux", feature = "webview-popover"))]
    gtk_ready: bool,
}

#[cfg(feature = "webview-popover")]
struct Popover {
    #[cfg(not(target_os = "linux"))]
    window: std::sync::Arc<dyn winit::window::Window>,
    #[cfg(target_os = "linux")]
    window: gtk::Window,
    webview: wry::WebView,
    width: f64,
    height: f64,
}

#[cfg(feature = "webview-popover")]
#[derive(serde::Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum PopoverMessage {
    Resize { width: f64, height: f64 },
    Drag,
    Move { dx: i32, dy: i32 },
    Close,
}

impl ApplicationHandler for TrayApp {
    fn can_create_surfaces(&mut self, _event_loop: &dyn ActiveEventLoop) {}

    fn resumed(&mut self, _event_loop: &dyn ActiveEventLoop) {}

    #[allow(unused_variables)]
    fn window_event(
        &mut self,
        _event_loop: &dyn ActiveEventLoop,
        id: WindowId,
        event: WindowEvent,
    ) {
        #[cfg(all(not(target_os = "linux"), feature = "webview-popover"))]
        {
            let is_popover = self
                .popover
                .as_ref()
                .is_some_and(|popover| popover.window.id() == id);
            if is_popover {
                match event {
                    WindowEvent::CloseRequested => {
                        self.close_popover();
                    }
                    // Click-outside-to-dismiss via focus loss. NOT on Windows: the
                    // WebView2 child window steals focus from the parent winit window as
                    // soon as it renders, which fires Focused(false) and would close the
                    // popover instantly (the "white flash then closes" bug). On Windows we
                    // dismiss via the tray-icon toggle, the in-page close button, or Quit.
                    #[cfg(not(windows))]
                    WindowEvent::Focused(false) => {
                        self.close_popover();
                    }
                    WindowEvent::Destroyed => {
                        self.popover.take();
                    }
                    _ => {}
                }
            }
        }
    }

    fn new_events(&mut self, event_loop: &dyn ActiveEventLoop, _cause: winit::event::StartCause) {
        pump_gtk_events();

        if self.tray.is_none() {
            let initial_metrics = self.tray_state.get_metrics();

            // Right-click context menu — present on every platform/config so there is
            // always a way to open the dashboard and quit, even when the WebView2
            // popover is unavailable or misbehaving.
            let menu = tray_icon::menu::Menu::new();
            let open_item = tray_icon::menu::MenuItem::new("Open Dashboard", true, None);
            let open_logs_item = tray_icon::menu::MenuItem::new("Open Logs Folder", true, None);
            let quit_item = tray_icon::menu::MenuItem::new("Quit Local LLM Foundry", true, None);
            let _ = menu.append(&open_item);
            let _ = menu.append(&open_logs_item);
            let _ = menu.append(&tray_icon::menu::PredefinedMenuItem::separator());
            let _ = menu.append(&quit_item);
            self.menu_open_id = Some(open_item.id().clone());
            self.menu_open_logs_id = Some(open_logs_item.id().clone());
            self.menu_quit_id = Some(quit_item.id().clone());

            let builder = TrayIconBuilder::new()
                .with_tooltip(crate::identity::PRODUCT_NAME)
                .with_menu(Box::new(menu))
                // Keep left-click for the popover toggle; menu is right-click only.
                .with_menu_on_left_click(false);

            let icon = std::mem::replace(
                &mut self.icon,
                Icon::from_rgba(vec![0, 0, 0, 255], 1, 1).unwrap(),
            );

            #[cfg(target_os = "macos")]
            let builder = builder.with_icon_templated(icon);

            #[cfg(not(target_os = "macos"))]
            let builder = builder.with_icon(icon);

            match builder.build() {
                Ok(tray) => {
                    self.tray = Some(tray);
                }
                Err(e) => {
                    eprintln!("[tray] Failed to create tray icon: {e}");
                    self.tray_start_failed.store(true, Ordering::Relaxed);
                    event_loop.exit();
                    return;
                }
            }

            if let Some(ref tooltip) = initial_metrics.3
                && let Some(ref tray) = self.tray
            {
                let _ = tray.set_tooltip(Some(tooltip));
            }

            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(500),
            ));
        }

        // Right-click context-menu events (Quit / Open Dashboard / Open Logs Folder).
        // Handled on every platform and feature configuration so the app is always quittable.
        while let Ok(menu_event) = tray_icon::menu::MenuEvent::receiver().try_recv() {
            if self.menu_quit_id.as_ref() == Some(&menu_event.id) {
                event_loop.exit();
            } else if self.menu_open_id.as_ref() == Some(&menu_event.id) {
                open_dashboard(self.port);
            } else if self.menu_open_logs_id.as_ref() == Some(&menu_event.id) {
                open_logs_folder(&self.app_root);
            }
        }

        #[cfg(feature = "webview-popover")]
        while let Ok(tray_event) = TrayIconEvent::receiver().try_recv() {
            match &tray_event {
                TrayIconEvent::Click {
                    button,
                    button_state,
                    position,
                    rect,
                    ..
                } if *button == tray_icon::MouseButton::Left
                    && *button_state == tray_icon::MouseButtonState::Down =>
                {
                    if self.popover.is_some() {
                        self.close_popover();
                    } else {
                        let rect = self.resolve_popover_anchor(event_loop, *rect, *position);
                        self.open_popover(event_loop, rect);
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        pump_gtk_events();
        self.refresh_tray_status();
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(500),
        ));
    }

    fn proxy_wake_up(&mut self, _event_loop: &dyn ActiveEventLoop) {
        #[cfg(feature = "webview-popover")]
        {
            let mut latest = None;
            let mut close_requested = false;
            while let Ok(message) = self.popover_rx.try_recv() {
                match message {
                    PopoverMessage::Resize { width, height } => {
                        latest = Some((width, height));
                    }
                    PopoverMessage::Drag =>
                    {
                        #[cfg(not(target_os = "linux"))]
                        if let Some(popover) = self.popover.as_ref() {
                            let _ = popover.window.drag_window();
                        }
                    }
                    PopoverMessage::Move { dx, dy } => {
                        #[cfg(not(target_os = "linux"))]
                        if let Some(popover) = self.popover.as_ref()
                            && let Ok(position) = popover.window.outer_position()
                        {
                            popover.window.set_outer_position(Position::Physical(
                                PhysicalPosition::new(
                                    position.x.saturating_add(dx),
                                    position.y.saturating_add(dy),
                                ),
                            ));
                        }
                        #[cfg(target_os = "linux")]
                        let _ = (dx, dy);
                    }
                    PopoverMessage::Close => close_requested = true,
                }
            }

            if close_requested {
                self.close_popover();
            } else if let Some((width, height)) = latest {
                self.resize_popover(width, height);
            }
        }
    }
}

impl TrayApp {
    fn refresh_tray_status(&mut self) {
        let metrics = self.tray_state.get_metrics();
        if let Some(ref tooltip) = metrics.3
            && let Some(ref tray) = self.tray
        {
            let _ = tray.set_tooltip(Some(tooltip));
        }
    }

    #[cfg(feature = "webview-popover")]
    fn open_popover(&mut self, event_loop: &dyn ActiveEventLoop, icon_rect: tray_icon::Rect) {
        let width = POPOVER_WIDTH;
        let height = POPOVER_INITIAL_HEIGHT;
        let (x, y) = popover_position(event_loop, icon_rect, width, height);

        let url = format!(
            "http://127.0.0.1:{}/compact?t={}",
            self.port,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        let proxy = event_loop.create_proxy();
        let popover_tx = self.popover_tx.clone();

        #[cfg(target_os = "linux")]
        {
            if !self.gtk_ready {
                return;
            }

            let window = gtk::Window::new(gtk::WindowType::Popup);
            window.set_decorated(false);
            window.set_resizable(false);
            window.set_keep_above(true);
            window.set_default_size(width as i32, height as i32);
            window.move_(x as i32, y as i32);

            let fixed = gtk::Fixed::new();
            fixed.set_size_request(width as i32, height as i32);
            window.add(&fixed);

            let webview = match wry::WebViewBuilder::new()
                .with_ipc_handler(move |request| {
                    if let Ok(message) = serde_json::from_str::<PopoverMessage>(request.body()) {
                        let _ = popover_tx.send(message);
                        proxy.wake_up();
                    }
                })
                .with_url(url)
                .with_bounds(wry::Rect {
                    position: wry::dpi::LogicalPosition::new(0.0, 0.0).into(),
                    size: wry::dpi::LogicalSize::new(width, height).into(),
                })
                .build_gtk(&fixed)
            {
                Ok(wv) => wv,
                Err(e) => {
                    eprintln!("[tray] Failed to create Linux tray WebView: {e}");
                    window.close();
                    return;
                }
            };

            window.show_all();
            self.popover = Some(Popover {
                window,
                webview,
                width,
                height,
            });
        }

        #[cfg(not(target_os = "linux"))]
        let attrs = WindowAttributes::default()
            .with_surface_size(winit::dpi::LogicalSize::new(width, height))
            .with_position(PhysicalPosition::new(x as i32, y as i32))
            .with_decorations(false)
            // Keep the Windows popover user-resizable. The compact page still
            // reports content-driven height through IPC, while this enables
            // normal edge/corner resizing from the borderless Winit window.
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_visible(true);

        #[cfg(not(target_os = "linux"))]
        let window: std::sync::Arc<dyn winit::window::Window> =
            match event_loop.create_window(attrs) {
                Ok(w) => std::sync::Arc::from(w),
                Err(_) => return,
            };

        #[cfg(not(target_os = "linux"))]
        let webview_builder = wry::WebViewBuilder::new()
            .with_ipc_handler(move |request| {
                if let Ok(message) = serde_json::from_str::<PopoverMessage>(request.body()) {
                    let _ = popover_tx.send(message);
                    proxy.wake_up();
                }
            })
            .with_url(url)
            .with_bounds(wry::Rect {
                position: wry::dpi::LogicalPosition::new(0.0, 0.0).into(),
                size: wry::dpi::LogicalSize::new(width, height).into(),
            });

        // Verify on real Windows hardware: confirm window.ipc.postMessage bridges via
        // WebView2; add with_initialization_script polyfill forwarding
        // window.chrome.webview.postMessage if messages don't arrive.
        #[cfg(not(target_os = "linux"))]
        let webview = match webview_builder.build_as_child(&window) {
            Ok(wv) => wv,
            Err(e) => {
                // On Windows the most common cause of build_as_child failure is a
                // missing WebView2 runtime (not present on LTSC / older Win10 images).
                // Surface a clear message and attempt a silent winget install so the
                // user can restart and get a working popover without manual steps.
                #[cfg(windows)]
                {
                    let err_str = e.to_string().to_lowercase();
                    let likely_missing_runtime = err_str.contains("webview2")
                        || err_str.contains("regdb")
                        || err_str.contains("0x800700c1")
                        || err_str.contains("class not registered")
                        || err_str.contains("co_e_classstring");
                    if likely_missing_runtime {
                        eprintln!(
                            "[tray] Tray popover unavailable: WebView2 runtime not found. \
                             Attempting automatic install — restart Local LLM Foundry afterward. \
                             Manual download: \
                             https://developer.microsoft.com/microsoft-edge/webview2/"
                        );
                        try_install_webview2();
                    } else {
                        eprintln!("[tray] Failed to create tray WebView: {e}");
                    }
                }
                #[cfg(not(windows))]
                eprintln!("[tray] Failed to create tray WebView: {e}");
                return;
            }
        };

        #[cfg(not(target_os = "linux"))]
        {
            self.popover = Some(Popover {
                window,
                webview,
                width,
                height,
            });
        }
    }

    #[cfg(feature = "webview-popover")]
    fn close_popover(&mut self) {
        if let Some(ref tray) = self.tray {
            let _ = tray.set_visible(true);
        }

        #[cfg(target_os = "linux")]
        if let Some(popover) = self.popover.take() {
            popover.window.close();
        }

        #[cfg(not(target_os = "linux"))]
        {
            self.popover.take();
        }
    }

    #[cfg(feature = "webview-popover")]
    fn resize_popover(&mut self, reported_width: f64, height: f64) {
        let Some(popover) = self.popover.as_mut() else {
            return;
        };

        let width = reported_width.clamp(POPOVER_MIN_WIDTH, POPOVER_MAX_WIDTH);
        let height = height.clamp(POPOVER_MIN_HEIGHT, POPOVER_MAX_HEIGHT);
        if (popover.width - width).abs() < 1.0 && (popover.height - height).abs() < 1.0 {
            return;
        }

        #[cfg(target_os = "linux")]
        {
            popover.window.resize(width as i32, height as i32);
            popover.window.set_default_size(width as i32, height as i32);
        }

        #[cfg(not(target_os = "linux"))]
        let _ = popover
            .window
            .request_surface_size(winit::dpi::LogicalSize::new(width, height).into());

        let _ = popover.webview.set_bounds(wry::Rect {
            position: wry::dpi::LogicalPosition::new(0.0, 0.0).into(),
            size: wry::dpi::LogicalSize::new(width, height).into(),
        });
        popover.width = width;
        popover.height = height;
    }

    #[cfg(feature = "webview-popover")]
    fn resolve_popover_anchor(
        &self,
        event_loop: &dyn ActiveEventLoop,
        event_rect: tray_icon::Rect,
        click_position: PhysicalPosition<f64>,
    ) -> tray_icon::Rect {
        if rect_has_position(event_rect) {
            return event_rect;
        }

        if let Some(ref tray) = self.tray
            && let Some(rect) = tray.rect()
            && rect_has_position(rect)
        {
            return rect;
        }

        if click_position.x > 0.0 || click_position.y > 0.0 {
            return tray_icon::Rect {
                position: PhysicalPosition::new(click_position.x - 11.0, click_position.y - 11.0),
                size: PhysicalSize::new(22, 22),
            };
        }

        if let Some(monitor) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
        {
            let monitor_pos = monitor
                .position()
                .unwrap_or_else(|| PhysicalPosition::new(0, 0));
            let monitor_size = monitor
                .current_video_mode()
                .map(|mode| mode.size())
                .unwrap_or_else(|| PhysicalSize::new(1024, 768));
            return tray_icon::Rect {
                position: PhysicalPosition::new(
                    monitor_pos.x as f64 + monitor_size.width as f64 - POPOVER_WIDTH - 16.0,
                    monitor_pos.y as f64 + 32.0,
                ),
                size: PhysicalSize::new(22, 22),
            };
        }

        event_rect
    }
}

#[cfg(feature = "webview-popover")]
fn rect_has_position(rect: tray_icon::Rect) -> bool {
    rect.size.width > 0 || rect.size.height > 0 || rect.position.x > 0.0 || rect.position.y > 0.0
}

/// Compute an on-screen position for the popover relative to the tray icon.
///
/// Defaults to just below the icon, but flips to *above* it when below would run off
/// the bottom of the monitor (the Windows taskbar is typically at the bottom, so the
/// tray icon sits low — opening downward clipped the popover off-screen). The result is
/// clamped so the whole popover stays within the monitor bounds. Coordinates are
/// physical pixels, matching `icon_rect.position` and the window `with_position` call.
#[cfg(feature = "webview-popover")]
fn popover_position(
    event_loop: &dyn ActiveEventLoop,
    icon_rect: tray_icon::Rect,
    width: f64,
    height: f64,
) -> (f64, f64) {
    let pos = icon_rect.position;
    let icon_w = icon_rect.size.width as f64;
    let icon_h = icon_rect.size.height as f64;

    let mut x = pos.x + (icon_w / 2.0) - (width / 2.0);
    let mut y = pos.y + icon_h + 4.0;

    // Prefer the monitor that contains the tray icon; fall back to primary/first.
    let monitor = event_loop
        .available_monitors()
        .find(|m| {
            let Some(mp) = m.position() else { return false };
            let Some(ms) = m.current_video_mode().map(|mode| mode.size()) else {
                return false;
            };
            let (left, top) = (mp.x as f64, mp.y as f64);
            pos.x >= left
                && pos.x < left + ms.width as f64
                && pos.y >= top
                && pos.y < top + ms.height as f64
        })
        .or_else(|| event_loop.primary_monitor());

    if let Some(monitor) = monitor
        && let Some(mp) = monitor.position()
        && let Some(ms) = monitor.current_video_mode().map(|mode| mode.size())
    {
        let left = mp.x as f64;
        let top = mp.y as f64;
        let right = left + ms.width as f64;
        let bottom = top + ms.height as f64;

        // Flip above the icon if opening below would overflow the bottom edge.
        if y + height > bottom {
            y = pos.y - height - 4.0;
        }
        // Keep the whole popover on-screen.
        x = x.clamp(left, (right - width).max(left));
        y = y.clamp(top, (bottom - height).max(top));
    }

    (x, y)
}

/// Open the dashboard in the user's default browser. Used by the tray context menu.
/// Routed through `no_window` on Windows so it never flashes a console.
fn open_dashboard(port: u16) {
    let url = format!("http://127.0.0.1:{port}/");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        // `explorer <url>` opens the default browser without a console window.
        let mut c = std::process::Command::new("explorer.exe");
        c.arg(&url);
        crate::platform::no_window(&mut c);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("open");
        c.arg(&url);
        c
    };
    #[cfg(target_os = "linux")]
    let mut cmd = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(&url);
        c
    };
    if let Err(e) = cmd.spawn() {
        eprintln!("[tray] failed to open dashboard {url}: {e}");
    }
}

/// Open the logs directory in the file explorer.
/// Uses:
/// - Windows: explorer
/// - macOS: open
/// - Linux: xdg-open
///
/// If the logs directory doesn't exist, creates it.
fn open_logs_folder(app_root: &std::path::Path) {
    let config_dir = app_root.to_path_buf();
    let logs_dir = crate::paths::AppPaths::from_root(config_dir.clone()).logs_dir();

    // Defensive: canonicalize and ensure it is still under config_dir
    // to guard against future config_dir overrides with .. segments.
    let Ok(canonical_config) = config_dir.canonicalize() else {
        eprintln!(
            "[tray] failed to open logs folder: could not canonicalize config dir {}",
            config_dir.display()
        );
        return;
    };
    if let Err(e) = std::fs::create_dir_all(&logs_dir) {
        eprintln!(
            "[tray] failed to create logs folder {}: {e}",
            logs_dir.display()
        );
        return;
    }
    let Some(canonical_logs) = logs_dir.canonicalize().ok() else {
        eprintln!(
            "[tray] failed to open logs folder: could not canonicalize logs dir {}",
            logs_dir.display()
        );
        return;
    };
    if !canonical_logs.starts_with(&canonical_config) {
        eprintln!(
            "[tray] failed to open logs folder: logs dir escaped config dir: {}",
            canonical_logs.display()
        );
        return;
    }

    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("explorer");
        c.arg(&logs_dir);
        crate::platform::no_window(&mut c);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("open");
        c.arg(&logs_dir);
        c
    };
    #[cfg(target_os = "linux")]
    let mut cmd = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(&logs_dir);
        c
    };

    if let Err(e) = cmd.spawn() {
        eprintln!(
            "[tray] failed to open logs folder {}: {e}",
            logs_dir.display()
        );
    }
}

#[cfg(all(target_os = "linux", feature = "webview-popover"))]
fn pump_gtk_events() {
    while gtk::events_pending() {
        gtk::main_iteration_do(false);
    }
}

#[cfg(not(all(target_os = "linux", feature = "webview-popover")))]
fn pump_gtk_events() {}

struct TrayState {
    app_state: Arc<AppState>,
}

impl TrayState {
    fn get_metrics(&self) -> TrayMetrics {
        let local_metrics_available = self.app_state.host_metrics_available();
        let sys = self.app_state.system_metrics.lock().unwrap().clone();
        let gpu = if local_metrics_available {
            self.app_state.gpu_metrics.lock().unwrap().clone()
        } else {
            Default::default()
        };
        let llama = self.app_state.llama_metrics.lock().unwrap().clone();
        let gpu_entries = if gpu.is_empty() {
            None
        } else {
            Some(gpu.into_iter().collect())
        };
        let tooltip = self.build_tooltip(&sys, &gpu_entries, &llama, local_metrics_available);
        (sys, gpu_entries, llama, Some(tooltip))
    }

    fn build_tooltip(
        &self,
        sys: &SystemMetrics,
        gpu: &Option<Vec<(String, GpuMetrics)>>,
        llama: &LlamaMetrics,
        local_metrics_available: bool,
    ) -> String {
        let mut lines = Vec::new();

        let endpoint_kind = self.app_state.current_endpoint_kind();
        let session_mode = match self.app_state.current_session_kind() {
            crate::state::SessionKind::Spawn => "Spawn",
            crate::state::SessionKind::Attach => "Attach",
            crate::state::SessionKind::None => "",
        };

        let local_label = if endpoint_kind == crate::state::EndpointKind::Local {
            "Local"
        } else {
            "Remote"
        };

        lines.push(format!("{} - {}", local_label, session_mode));

        if local_metrics_available {
            lines.push(format!("CPU: {}%", sys.cpu_load as f32 / 10.0));

            if sys.cpu_temp_available {
                lines.push(format!("Temp: {:.0}C", sys.cpu_temp));
            }

            if let Some(g) = gpu {
                for (name, m) in g.iter() {
                    let vram_pct = if m.vram_total > 0 {
                        (m.vram_used as f64 / m.vram_total as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    lines.push(format!("{}: {:.0}C / {}% VRAM", name, m.temp, vram_pct));
                }
            }
        } else {
            lines.push("Host metrics unavailable".to_string());
        }

        if llama.generation_tokens_per_sec > 0.0 {
            lines.push(format!("{:.0} tok/s", llama.generation_tokens_per_sec));
        }

        lines.join("\n")
    }
}

#[cfg(all(test, feature = "webview-popover"))]
mod tests {
    use super::PopoverMessage;

    #[test]
    fn parses_popover_resize_message() {
        let message: PopoverMessage =
            serde_json::from_str(r#"{"action":"resize","width":240,"height":180}"#).unwrap();

        match message {
            PopoverMessage::Resize { width, height } => {
                assert_eq!(width, 240.0);
                assert_eq!(height, 180.0);
            }
            PopoverMessage::Drag => panic!("expected resize message"),
            PopoverMessage::Move { .. } => panic!("expected resize message"),
            PopoverMessage::Close => panic!("expected resize message"),
        }
    }

    #[test]
    fn parses_popover_close_message() {
        let message: PopoverMessage = serde_json::from_str(r#"{"action":"close"}"#).unwrap();

        assert!(matches!(message, PopoverMessage::Close));
    }
}
