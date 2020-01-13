$(document).ready(function () {
   
    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Hide btn
        $(this).hide();

    });

});
