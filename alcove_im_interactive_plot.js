$(document).ready(function() {
    // add listeners to each select dropdown to load a new figure
    $('.custom-select').on('change', function() {
        var net = $('#net-select').val()
        var loss_type = $('#loss-select').val()
        var image_set = $('#image-set-select').val()
        var lr_attention = $('#lr-attention-select').val()
        var lr_assoc = $('#lr-assoc-select').val()
        var c = $('#c-select').val()
        var phi = $('#phi-select').val()

        // update all model plots simultaneously
        var model_fig_filename = `img/alcove-vgg11-${loss_type}-${image_set}-${lr_attention}-${lr_assoc}-${c}-${phi}.svg`
        $('#model-fig-vgg-11').attr('src', model_fig_filename);

        var model_fig_filename = `img/alcove-resnet18-${loss_type}-${image_set}-${lr_attention}-${lr_assoc}-${c}-${phi}.svg`
        $('#model-fig-resnet-18').attr('src', model_fig_filename);

        var model_fig_filename = `img/alcove-resnet152-${loss_type}-${image_set}-${lr_attention}-${lr_assoc}-${c}-${phi}.svg`
        $('#model-fig-resnet-152').attr('src', model_fig_filename);
        
        console.log(model_fig_filename);
    });

});
