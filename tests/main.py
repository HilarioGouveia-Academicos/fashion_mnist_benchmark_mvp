import io
import os
import threading
import cv2
import numpy as np
import requests

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.widget import Widget
from kivy.utils import platform

# AJUSTE: Substitua pelo IP da sua máquina na rede local para testes no celular
API_BASE_URL = "http://192.168.0.15:8000"


class DrawingCanvas(Widget):
    """Área para desenhar uma peça de roupa (Fashion-MNIST)."""

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            with self.canvas:
                # Fashion-MNIST costuma ter fundo preto e traço branco
                Color(1, 1, 1)
                touch.ud["line"] = Line(points=(touch.x, touch.y), width=6)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if "line" in touch.ud:
            touch.ud["line"].points += [touch.x, touch.y]
            return True
        return super().on_touch_move(touch)

    def clear(self):
        self.canvas.clear()


class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=8, padding=10, **kwargs)

        # 1. Área de Exibição / Entrada
        self.display_area = BoxLayout(size_hint=(1, 0.65))
        self.drawing_widget = DrawingCanvas()
        self.camera_widget = None
        self.display_area.add_widget(self.drawing_widget)
        self.add_widget(self.display_area)

        # 2. Seletor de Modelo da API
        model_bar = BoxLayout(size_hint=(1, 0.08), spacing=8)
        model_bar.add_widget(Label(text="Modelo:", size_hint_x=0.3))
        self.model_spinner = Spinner(
            text="cnn",
            values=("cnn", "mlp", "svm"),
            size_hint_x=0.7,
        )
        model_bar.add_widget(self.model_spinner)
        self.add_widget(model_bar)

        # 3. Status e Resultado
        self.status_label = Label(
            text="Desenhe, fotografe ou carregue uma imagem.",
            size_hint=(1, 0.1),
            font_size="15sp",
        )
        self.add_widget(self.status_label)

        # 4. Barra de Controles
        self.build_controls()

    def build_controls(self):
        controls = BoxLayout(size_hint=(1, 0.17), spacing=6)

        btn_cam = Button(text="Câmera", on_press=self.toggle_camera)
        btn_draw = Button(text="Desenhar", on_press=self.toggle_drawing)
        btn_load = Button(text="Carregar", on_press=self.open_file_chooser)
        btn_predict = Button(
            text="Enviar p/ API",
            background_color=(0.1, 0.6, 0.3, 1),
            on_press=self.start_pipeline,
        )

        controls.add_widget(btn_cam)
        controls.add_widget(btn_draw)
        controls.add_widget(btn_load)
        controls.add_widget(btn_predict)
        self.add_widget(controls)

    # --- Alternância de Modos ---

    def toggle_drawing(self, instance):
        self.display_area.clear_widgets()
        if self.camera_widget:
            self.camera_widget.play = False
            self.camera_widget = None
        self.drawing_widget.clear()
        self.display_area.add_widget(self.drawing_widget)
        self.status_label.text = "Modo desenho ativo."

    def toggle_camera(self, instance):
        self.display_area.clear_widgets()
        if not self.camera_widget:
            self.camera_widget = Camera(play=True, resolution=(640, 480))
        self.display_area.add_widget(self.camera_widget)
        self.status_label.text = "Câmera ativa."

    def open_file_chooser(self, instance):
        content = BoxLayout(orientation="vertical")
        chooser = FileChooserIconView(filters=["*.png", "*.jpg", "*.jpeg"])
        btn_select = Button(text="Selecionar", size_hint_y=0.15)

        content.add_widget(chooser)
        content.add_widget(btn_select)

        popup = Popup(title="Escolha uma imagem", content=content, size_hint=(0.9, 0.9))

        def _select(btn_instance):
            if chooser.selection:
                file_path = chooser.selection[0]
                popup.dismiss()
                self.load_image_from_disk(file_path)

        btn_select.bind(on_press=_select)
        popup.open()

    def load_image_from_disk(self, file_path):
        self.display_area.clear_widgets()
        if self.camera_widget:
            self.camera_widget.play = False
            self.camera_widget = None
        preview = Image(source=file_path)
        self.display_area.add_widget(preview)
        self.display_area.loaded_path = file_path
        self.status_label.text = f"Carregado: {os.path.basename(file_path)}"

    # --- Captura da Imagem ---

    def acquire_raw_image(self):
        # 1. Da Câmera
        if self.camera_widget and self.camera_widget.parent:
            texture = self.camera_widget.texture
            size = texture.size
            arr = np.frombuffer(texture.pixels, dtype=np.uint8).reshape((size[1], size[0], 4))
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return cv2.flip(img_bgr, 0)

        # 2. Do Arquivo Selecionado
        if hasattr(self.display_area, "loaded_path") and os.path.exists(
            self.display_area.loaded_path
        ):
            return cv2.imread(self.display_area.loaded_path)

        # 3. Do Desenho no Canvas
        temp_path = "temp_drawing.png"
        self.drawing_widget.export_to_png(temp_path)
        img = cv2.imread(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return img

    def encode_image(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Nenhuma imagem capturada.")
        # Redimensiona para formato padrão leve antes de trafegar na rede
        resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".png", resized)
        return buffer.tobytes()

    # --- Requisição à FastAPI ---

    def start_pipeline(self, instance):
        self.status_label.text = "Enviando para o servidor..."
        selected_model = self.model_spinner.text
        threading.Thread(
            target=self._worker_request, args=(selected_model,), daemon=True
        ).start()

    def _worker_request(self, model_name):
        try:
            raw_img = self.acquire_raw_image()
            payload = self.encode_image(raw_img)

            url = f"{API_BASE_URL}/predict?model={model_name}"
            files = {"file": ("input.png", io.BytesIO(payload), "image/png")}

            response = requests.post(url, files=files, timeout=8)

            if response.status_code == 200:
                data = response.json()
                # Adapta dinamicamente ao formato que seu Predictor retornar
                pred = data.get("prediction", data.get("label", str(data)))
                msg = f"Resultado [{model_name}]: {pred}"
            else:
                msg = f"Erro {response.status_code}: {response.text}"

        except Exception as err:
            msg = f"Falha na rede: {str(err)[:40]}"

        Clock.schedule_once(lambda dt: self._update_ui(msg), 0)

    def _update_ui(self, text):
        self.status_label.text = text


class MobileClientApp(App):
    def build(self):
        return MainLayout()

    def on_start(self):
        # Permissões em tempo de execução no Android
        if platform == "android":
            from android.permissions import Permission, request_permissions

            request_permissions(
                [
                    Permission.CAMERA,
                    Permission.READ_EXTERNAL_STORAGE,
                    Permission.WRITE_EXTERNAL_STORAGE,
                    Permission.INTERNET,
                ]
            )


if __name__ == "__main__":
    MobileClientApp().run()