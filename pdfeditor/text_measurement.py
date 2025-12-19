def _measure_text_height(
    width: float,
    text: str,
    fontname: str,
    fontsize: float,
    align: int = 0
) -> float:
    """
    Measure height required for text in a given width using a temporary page.
    """
    # Create a temp doc/page large enough
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page(width=1000, height=5000)
    
    # Define a rect with target width and ample height
    # x0=0 is fine for measurement
    rect = fitz.Rect(0, 0, width, 5000)
    
    rc = temp_page.insert_textbox(
        rect,
        text,
        fontname=fontname,
        fontsize=fontsize,
        align=align
    )
    
    temp_doc.close()
    
    if rc < 0:
        # Should not happen with height=5000 unless text is massive
        return 5000.0
        
    # rc is unused vertical space
    used_height = 5000.0 - rc
    return used_height
