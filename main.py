from GrammerChecker import *                                        # Import necessary modules
from SpellingChecker import *
import tkinter as tk
from tkinter import ttk, messagebox
import time
import random
import nltk
import logging                                                      #For logging application events

logging.basicConfig(                                                #configure logging format and level
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class GrammarSpellCheckerApp:
    def __init__(self, root):
        #Initialize the main application window and components.
        logging.info("Starting Grammar & Spelling Checker App")
        self.root = root                                            #main window configuration
        self.root.title("English Grammar & Spelling Checker")
        self.root.geometry("1200x700")
        self.colors = {                                             #color scheme for the application
            'bg': '#1e1e1e',
            'bg_secondary': '#2b2b2b',
            'bg_tertiary': '#252525',
            'text': '#e0e0e0',
            'text_secondary': '#a0a0a0',
            'highlight': '#4CAF50',
            'highlight_dark': '#388E3C',
            'highlight_light': '#81C784',
            'error': '#FF5252',
            'warning': '#FFC107',
            'success': '#4CAF50',
            'widget_bg': '#333333',
            'widget_highlight': '#4CAF50',
            'widget_border': '#3e3e3e',
            'suggestion_bg': '#2b2b2b',
            'suggestion_highlight': '#4CAF50',
        }

        self.root.configure(bg=self.colors['bg'])                   #Configure root window background
        self.style = ttk.Style()
        self.style.theme_use('clam')
        #configure the ttk widget styles for consistent appearance.
        self.style.configure('.', background=self.colors['bg'], foreground=self.colors['text'])
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'])
        self.style.configure('TButton',                             # Button style configuration
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text'],
                             bordercolor=self.colors['widget_border'],
                             lightcolor=self.colors['bg_secondary'],
                             darkcolor=self.colors['bg_secondary'],
                             relief='flat')
        self.style.map('TButton',                                   # Button active states
                       background=[('active', self.colors['highlight_dark']),
                                   ('pressed', self.colors['highlight'])],
                       foreground=[('active', 'white')])
        self.spell_checker = SpellingChecker()                      #Initialize checkers  
        logging.info("SpellingChecker initialized")
        self.grammar_checker = GrammarChecker()
        logging.info("GrammarChecker initialized")
        
        corpus = 'Cleaned_Lang8.csv'                                # train grammar models
        self.grammar_checker.train_models(corpus)
        logging.info(f"Trained grammar models using corpus: {corpus}")

        self.create_widgets()                                       #Create GUI widgets
        logging.info("GUI widgets created")
        logging.info("Application fully initialized and ready.")
        # Initialize application state variables
        self.current_errors = []                                    #stores current detected errors
        self.current_text = ""                                      #stores current text content
        self.debounce_id = None                                     # For delaying check_text()

        # Final initialization check
        self.root.after(100, lambda: logging.info("GUI is ready and displayed."))

    #Create and arrange all GUI components.
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)                           # Main container frame
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)                          #left panel (text input area)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        input_header = ttk.Frame(left_frame)                        #input text header
        input_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(input_header, text="Input Text", font=("Segoe UI", 12, 'bold')).pack(side=tk.LEFT)

        self.text_frame = ttk.Frame(left_frame)                     #text input area with line numbers
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        #Line numbers widget
        self.line_numbers = tk.Text(self.text_frame, width=4, padx=4, pady=4, takefocus=0, bd=0,
                                    bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                    state='disabled', font=("Segoe UI", 10))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        #USER text input widget
        self.input_text = tk.Text(self.text_frame, wrap=tk.WORD, font=("Segoe UI", 12), 
                                  bg=self.colors['widget_bg'], fg=self.colors['text'],
                                  height=30, width=50, bd=0, highlightthickness=1, 
                                  highlightbackground=self.colors['widget_border'], 
                                  highlightcolor=self.colors['widget_highlight'],
                                  insertbackground=self.colors['highlight'],
                                  selectbackground=self.colors['highlight_light'],
                                  selectforeground='white',
                                  padx=10, pady=10)
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_text.bind('<KeyRelease>', self.on_text_change)
        self.input_text.bind('<KeyPress>', self.update_line_numbers)

        right_frame = ttk.Frame(main_frame)                         #right panel (controls and correction suggestions)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        controls_frame = ttk.Frame(right_frame)                     # Controls frame
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        #GRAMMER MODEL selection
        model_frame = ttk.LabelFrame(controls_frame, text="Grammar Model", padding=10)
        model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.model_var = tk.StringVar(value="rule_based")
        ttk.Radiobutton(model_frame, text="Rule-based", variable=self.model_var, 
                        value="rule_based", command=self.check_text).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Statistical", variable=self.model_var, 
                        value="statistical", command=self.check_text).pack(side=tk.LEFT, padx=5)

        self.style.configure('TNotebook', background=self.colors['bg'])
        self.style.configure('TNotebook.Tab', 
                             background=self.colors['bg_secondary'],
                             foreground=self.colors['text'],
                             padding=[10, 5],
                             font=('Segoe UI', 9))
        self.style.map('TNotebook.Tab',
                       background=[('selected', self.colors['highlight'])],
                       foreground=[('selected', 'white')],
                       expand=[('selected', [1, 1, 1, 0])])

        filter_notebook = ttk.Notebook(right_frame, style='TNotebook')
        filter_notebook.pack(fill=tk.X, pady=(0, 10))               #configure notebook style for error filtering tabs
        all_tab = ttk.Frame(filter_notebook)                        # Create tabs for different error types
        grammar_tab = ttk.Frame(filter_notebook)                    #=> 'All' / 'Grammer' / 'Spelling'
        spelling_tab = ttk.Frame(filter_notebook)
        filter_notebook.add(all_tab, text=" ALL ")
        filter_notebook.add(grammar_tab, text=" GRAMMAR ")
        filter_notebook.add(spelling_tab, text=" SPELLING ")
        self.filter_notebook = filter_notebook
        filter_notebook.bind("<<NotebookTabChanged>>", lambda e: self.update_suggestions())

        separator = ttk.Separator(right_frame, orient='horizontal') #separator
        separator.pack(fill=tk.X, pady=(0, 10))

        suggestions_header = ttk.Frame(right_frame)                 #suggestions header
        suggestions_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(suggestions_header, text="Suggestions", font=("Segoe UI", 12, 'bold')).pack(side=tk.LEFT)
        self.stats_label = ttk.Label(suggestions_header, text="0 errors found")
        self.stats_label.pack(side=tk.RIGHT)

        self.suggestions_container = ttk.Frame(right_frame)         #Suggestions display area with scrollbar
        self.suggestions_container.pack(fill=tk.BOTH, expand=True)
        self.suggestions_canvas = tk.Canvas(self.suggestions_container, 
                                            bg=self.colors['bg_secondary'],
                                            bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.suggestions_container,  #Scrollbar
                                       orient="vertical", 
                                       command=self.suggestions_canvas.yview)
        self.suggestions_frame = ttk.Frame(self.suggestions_canvas)
        self.suggestions_frame.bind(                               #Frame that holds the suggestions 
            "<Configure>",                  
            lambda e: self.suggestions_canvas.configure(
                scrollregion=self.suggestions_canvas.bbox("all")
            )
        )
        # Configure canvas scrolling
        self.suggestions_canvas.create_window((0, 0), window=self.suggestions_frame, anchor="nw")
        self.suggestions_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.suggestions_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.suggestions_canvas.bind_all("<MouseWheel>", self._on_mousewheel)# Mouse wheel support for scrolling

        self.status_bar = ttk.Frame(self.root, height=20)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = ttk.Label(self.status_bar, text="Ready", background=self.colors['bg_secondary'])
        self.status_label.pack(fill=tk.X, padx=1, pady=1)

    def update_line_numbers(self, event=None):
        #Update the line numbers in the sidebar to match the text content.
        lines = self.input_text.get('1.0', tk.END).count('\n')
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', tk.END)
        for i in range(1, lines + 1):
            self.line_numbers.insert(tk.END, f"{i}\n")
        self.line_numbers.config(state='disabled')

    def _on_mousewheel(self, event):
        #Handle mouse wheel scrolling for the suggestions canvas.
        self.suggestions_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_text_change(self, event=None):
        #Handle text changes in the input area with debounce.
        self.current_text = self.input_text.get("1.0", tk.END)
        self.update_line_numbers()
        self.clear_highlights()
        self.status_label.config(text="Waiting to check...")

        if self.debounce_id is not None:                            #cancel any pending checks
            self.root.after_cancel(self.debounce_id)
        self.debounce_id = self.root.after(2000, self.check_text)   #Only check for errors after 2s of no keyboard input

    def clear_highlights(self):
        #Remove all error highlights from the text widget.
        self.input_text.tag_remove("spelling_error", "1.0", tk.END)
        self.input_text.tag_remove("grammar_error", "1.0", tk.END)
        self.input_text.tag_remove("highlight", "1.0", tk.END)

    def highlight_errors(self):
        #highlight detected errors in the text widget.
        self.clear_highlights()
        for error in self.current_errors:                           #apply tags to each error in the text
            start = "1.0"
            while True:
                pos = self.input_text.search(error['error'], start, tk.END)
                if not pos:
                    break
                end = f"{pos}+{len(error['error'])}c"
                tag_name = f"{error['type']}_error"
                self.input_text.tag_add(tag_name, pos, end)
                start = end
        # Configure tag appearance
        self.input_text.tag_config("spelling_error", background=self.colors['bg_secondary'], underline=True, underlinefg=self.colors['error'])
        self.input_text.tag_config("grammar_error", background=self.colors['bg_secondary'], underline=True, underlinefg=self.colors['warning'])
        #update error statistics
        grammar_count = len([e for e in self.current_errors if e['type'] == 'grammar'])
        spelling_count = len([e for e in self.current_errors if e['type'] == 'spelling'])
        self.stats_label.config(text=f"{len(self.current_errors)} errors ({grammar_count} grammar, {spelling_count} spelling)")
        self.status_label.config(text="Ready")

    def check_text(self):
        #Check the text for grammar and spelling errors.
        logging.info("Checking text for grammar and spelling errors...")
        text = self.input_text.get("1.0", tk.END)
        if not text.strip():                                        #skip check if text is empty
            self.current_errors = []
            self.update_suggestions()
            return

        spelling_errors = self.spell_checker.check_spelling(text)   #perform spelling and grammar checks
        if self.model_var.get() == "rule_based":                    #Use selected grammar model based on USER selection
            grammar_errors = self.grammar_checker.rule_based_check(text)
        else:
            grammar_errors = self.grammar_checker.statistical_check(text)

        self.current_errors = spelling_errors + grammar_errors      #Update current errors and UI
        self.highlight_errors()
        self.update_suggestions()

    def update_suggestions(self, event=None):
        #Update the suggestions panel based on current errors and selected tab.
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()                                        #CLEAR existing suggestions
        if not self.current_errors:
            no_errors_frame = ttk.Frame(self.suggestions_frame)
            no_errors_frame.pack(fill=tk.X)                         #Show message if no errors
            ttk.Label(no_errors_frame, text="No errors detected",
                      foreground=self.colors['highlight'],
                      background=self.colors['bg_secondary'],
                      font=("Segoe UI", 15, 'bold')).pack()
            return
        # Determine which errors to show based on selected tab
        current_tab = self.filter_notebook.index(self.filter_notebook.select())
        if current_tab == 0:                                        #ALL errors
            errors_to_show = self.current_errors
        elif current_tab == 1:                                      #GRAMMER errors only
            errors_to_show = [e for e in self.current_errors if e['type'] == 'grammar']
        else:                                                       #SPELLING errors only
            errors_to_show = [e for e in self.current_errors if e['type'] == 'spelling']

        if not errors_to_show:                                      #show message if no errors in current category
            no_filtered_frame = ttk.Frame(self.suggestions_frame)
            no_filtered_frame.pack(fill=tk.X)
            tab_names = ["All", "Grammar", "Spelling"]
            ttk.Label(no_filtered_frame, text=f"No {tab_names[current_tab].lower()} errors detected",
                      foreground=self.colors['highlight'],
                      background=self.colors['bg_secondary'],
                      font=("Segoe UI", 15, 'bold')).pack()
            return

        for error in errors_to_show:                                #display each error with suggestions
            frame = ttk.Frame(self.suggestions_frame, style='TFrame')
            frame.pack(fill=tk.X, padx=5, pady=5)
            #error suggestion container
            suggestion_frame = ttk.Frame(frame, style='TFrame', relief=tk.GROOVE, borderwidth=1, padding=10)
            suggestion_frame.pack(fill=tk.X)
            #error type indicator
            type_color = self.colors['error'] if error['type'] == 'spelling' else self.colors['warning']
            type_frame = ttk.Frame(suggestion_frame, style='TFrame')
            type_frame.pack(fill=tk.X, pady=(0, 5))
            ttk.Label(type_frame, text=error['type'].capitalize(), foreground=type_color, font=("Segoe UI", 9, 'bold')).pack(side=tk.LEFT)
            # Error message
            msg_frame = ttk.Frame(suggestion_frame, style='TFrame')
            msg_frame.pack(fill=tk.X, pady=(0, 10))
            ttk.Label(msg_frame, text=error['message'], wraplength=400, justify=tk.LEFT).pack(anchor='w')
            
            # Error display with wrapping
            error_frame = ttk.Frame(suggestion_frame, style='TFrame')
            error_frame.pack(fill=tk.X, pady=(0, 5))
            ttk.Label(error_frame, text="Error:", font=("Segoe UI", 10), 
                    width=8, anchor='w').pack(side=tk.LEFT, anchor='n')
            error_text_frame = ttk.Frame(error_frame)
            error_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(error_text_frame, text=error['error'], font=("Segoe UI", 12, 'bold'),
                    wraplength=400, justify=tk.LEFT).pack(anchor='w')
            
            # Correction display with wrapping
            correction_frame = ttk.Frame(suggestion_frame, style='TFrame')
            correction_frame.pack(fill=tk.X, pady=(0, 5))
            ttk.Label(correction_frame, text="→", 
                    foreground=self.colors['highlight']).pack(side=tk.LEFT, padx=5)
            ttk.Label(correction_frame, text="Correction:", font=("Segoe UI", 10), 
                    width=10, anchor='w').pack(side=tk.LEFT)
            correction_text_frame = ttk.Frame(correction_frame)
            correction_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(correction_text_frame, text=error['correction'], 
                    font=("Segoe UI", 12, 'bold'), foreground=self.colors['success'],
                    wraplength=400, justify=tk.LEFT).pack(anchor='w')
            
            # Correction button
            btn_frame = ttk.Frame(suggestion_frame, style='TFrame')
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame,                       
                       text="Apply Correction",
                       command=lambda e=error: self.apply_correction(e),
                       style='TButton').pack(side=tk.RIGHT)

    def apply_correction(self, error):
        #Apply the selected correction to the text
        logging.info(f"Applied correction: {error['error']} -> {error['correction']}")
        text = self.input_text.get("1.0", tk.END)                   #Apply the selected correction to the text
        new_text = text.replace(error['error'], error['correction'])
        self.input_text.delete("1.0", tk.END)                       # Update text widget
        self.input_text.insert("1.0", new_text)
        # Update status and trigger re-check
        self.status_label.config(text=f"Applied correction: {error['error']} → {error['correction']}")
        self.on_text_change()

if __name__ == "__main__":
    root = tk.Tk()
    app = GrammarSpellCheckerApp(root)
    root.mainloop()
