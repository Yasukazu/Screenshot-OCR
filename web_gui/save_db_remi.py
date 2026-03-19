import remi.gui as gui
from remi import start, App
import sqlite3

class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def main(self):
        self.container = gui.VBox(width=300, height=300)
        
        # User input fields
        self.text_input = gui.TextInput(width=200, height=30)
        self.btn_save = gui.Button("Save to DB", width=200, height=30)
        
        # Listener for the onclick event [7]
        self.btn_save.onclick.do(self.on_save_pressed)
        
        self.container.append(self.text_input)
        self.container.append(self.btn_save)
        
        return self.container

    def on_save_pressed(self, widget):
        data = self.text_input.get_value()
        
        # Database operation [1, 9]
        conn = sqlite3.connect("app_data.db")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS user_data (content TEXT)")
        cursor.execute("INSERT INTO user_data (content) VALUES (?)", (data,))
        conn.commit()
        conn.close()
        
        self.text_input.set_value("Saved!")

if __name__ == "__main__":
    start(MyApp)
