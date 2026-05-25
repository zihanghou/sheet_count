import sys

from PySide6.QtWidgets import QApplication

from sheet_count_ui.main_window import SheetCountMainWindow


APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei", "SimSun", "Arial";
    font-size: 12pt;
}
QPushButton {
    background-color: #0078D7;
    color: white;
    border-radius: 6px;
    padding: 12px;
    min-height: 48px;
    font-size: 13pt;
}
QPushButton:hover {
    background-color: #005A9E;
}
QPushButton:checked {
    background-color: #004C87;
}
QPushButton:disabled {
    background-color: #888;
}
QLineEdit, QComboBox {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: 10px;
    min-height: 38px;
    font-size: 12pt;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = SheetCountMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
