from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget


class DetectionWidget(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(24)

        title = QLabel("检测流程")
        title.setStyleSheet("font-size: 20pt; font-weight: 700;")
        layout.addWidget(title)

        self.count_result_label = QLabel("--")
        self.count_result_label.setAlignment(Qt.AlignCenter)
        self.count_result_label.setMinimumHeight(180)
        self.count_result_label.setStyleSheet(
            """
            QLabel {
                background-color: #111827;
                color: #FDE047;
                border-radius: 8px;
                font-size: 76pt;
                font-weight: 800;
            }
            """
        )
        layout.addWidget(self.count_result_label)

        self.count_caption_label = QLabel("当前计数结果")
        self.count_caption_label.setAlignment(Qt.AlignCenter)
        self.count_caption_label.setStyleSheet("font-size: 16pt; color: #4B5563;")
        layout.addWidget(self.count_caption_label)

        self.import_btn = QPushButton("导入参数 JSON")
        self.import_btn.clicked.connect(self.main_window.import_params_json)

        self.detect_btn = QPushButton("采图并检测")
        self.detect_btn.clicked.connect(self.main_window.capture_count_result)

        self.auto_btn = QPushButton("开启自动检测")
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self.main_window.toggle_auto_detect)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(14)
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.detect_btn)
        button_layout.addWidget(self.auto_btn)
        layout.addLayout(button_layout)

        self.auto_status_label = QLabel("自动检测: 未开启")
        self.auto_status_label.setWordWrap(True)
        self.auto_status_label.setStyleSheet(
            "font-size: 14pt; font-weight: 600; color: #374151; padding: 6px 0;"
        )
        layout.addWidget(self.auto_status_label)

        self.param_label = QLabel("参数: 未导入")
        self.param_label.setWordWrap(True)
        self.param_label.setStyleSheet("font-size: 16pt; font-weight: 600; color: #111827; padding: 10px 0;")
        layout.addWidget(self.param_label)
        layout.addStretch(1)

    def set_count_result(self, count):
        self.count_result_label.setText("--" if count is None else str(count))

    def set_auto_running(self, running):
        self.auto_btn.blockSignals(True)
        self.auto_btn.setChecked(bool(running))
        self.auto_btn.setText("关闭自动检测" if running else "开启自动检测")
        self.auto_btn.blockSignals(False)

    def set_auto_status(self, text):
        self.auto_status_label.setText(text)
