
if(window.File) {
    var drop_file_obj = document.getElementById('prediction_input_drop_file');
    //var plot_prediction_input_div_obj = document.getElementById('plot_prediction_input_div');
    var output = document.getElementById('prediction_output');

    // drag event
    drop_file_obj.addEventListener('dragover', function(e) {
        // not to open
        e.stopPropagation();
        e.preventDefault();
    }, false);

    // drop event
    drop_file_obj.addEventListener('drop', function(e) {
        $("#operation-body").addClass('loading-spinner');

        // not to open
        e.stopPropagation();
        e.preventDefault();

        // get file data
        var fileData = e.dataTransfer.files;

        for (var i = 0; i < fileData.length; i++) {
            var fileToRead = fileData[i];
            console.log(fileToRead.type)
            // check the file type
            if(!fileToRead.type.match(/csv/)) {
                alert('Please select csv file');
                return;
            }

            var reader = new FileReader();
            // Read file into memory as UTF-8
            reader.readAsText(fileToRead);
            // Error
            reader.onerror = errorHandler
            // Success
            reader.onload = loadHandler;

            function loadHandler(evt) {
                var read_csv = evt.target.result;
                plot_input_data(read_csv);

                var div = document.createElement('div');
                var insert = '<p>filename: ' + fileToRead.name + '</p>';
                div.innerHTML = insert;
                $('#plot_prediction_input_div').append(div);
                // output.appendChild(div);

            }

            function errorHandler(evt) {
              if(evt.target.error.name == "NotReadableError") {
                  alert("Canno't read file");
              }
            }

        }
        $("#operation-body").removeClass('loading-spinner');

    }, false);


}

function conv_csv_to_array(csv) {
    var allTextLines = csv.split(/\r\n|\n/);
    var lines = [];
    for (var i = 0; i < allTextLines.length; i++) {
        var data = allTextLines[i].split(',');
            var tarr = [];
            for (var j=0; j<data.length; j++) {
                tarr.push(data[j]);
            }
            lines.push(tarr);
    }
    return lines
}

function convert_csv_to_plotly_data(csv, value_col_name='value') {
    var data = conv_csv_to_array(csv);
    console.log('csv data:' + data);


    var _labels = [], _value_data = [];
    for (var row in data) {
        _labels.push(data[row][0])
        _value_data.push(data[row][1])
    };

    var value_data = {
      type: 'line',
      name: value_col_name,
      x: _labels,
      y: _value_data,
      mode: 'lines+markers'
    };


    console.log('value_data:' + value_data)

    return [value_data];
}


function plot_input_data(csv) {
    var char_tag_id = 'plot_prediction_input'
    if (!(document.getElementById(char_tag_id) instanceof Object)) {
        $('#plot_prediction_input_div').append('<div id=\"' + char_tag_id + '\"></div>')
    }

    var plotly_data = convert_csv_to_plotly_data(csv);
    console.log('plotly_data:' + plotly_data)
    draw_with_plotly('input data', plotly_data, char_tag_id);
}