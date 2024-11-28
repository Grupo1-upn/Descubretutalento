// Al hacer clic en el botón para pasar a la sección de aptitudes
document.getElementById("nextAptitudes").addEventListener("click", function() {
    // Oculta la sección de intereses
    document.getElementById("section-intereses").style.display = "none";

    // Muestra la sección de aptitudes
    document.getElementById("section-aptitudes").style.display = "block";

    // Desplazar la página hasta la parte superior de la nueva sección
    window.scrollTo({
        top: document.getElementById("section-aptitudes").offsetTop, 
        behavior: "smooth" // Agrega un desplazamiento suave
    });
});
