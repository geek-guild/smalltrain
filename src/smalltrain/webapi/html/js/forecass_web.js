$('.tab_li').on('click', function(){
    console.log('clicked:' + $(this).attr('class'));
    console.log('disp for:' + $(this).attr('for'));

    $('.tab_li').removeClass('is-active');
    $(this).addClass('is-active');

    var currentAttrValue = $(this).attr('for');
    console.log('currentAttrValue:' + currentAttrValue)
    // Show/Hide Tabs
    $('.tabs_contents #' + currentAttrValue).show().siblings().hide();

});

function toggle_elements_by_read_status(have_read) {
    if (have_read) {
        // enale buttons
        document.getElementById('prediction_submit').disabled = false
        document.getElementById('cancel_submit').disabled = false
        // show operation info and setting tab selected
        $('#operation-body').show()
        $('.tabs_contents #tabs_setting').show().siblings().hide();
    } else {
        document.getElementById('prediction_submit').disabled = true
        document.getElementById('cancel_submit').disabled = true
        $('#operation-body').hide()
    }
}

// have not read initially
toggle_elements_by_read_status(have_read=false)

// setting host of API server
var host_url = '.' // for local

// Object of the form
var operation_read_form = document.getElementById('operation_read_form');
var history_read_form = document.getElementById('history_read_form');

/*
 * submit function that from the form and sends it to the server with ajax.
 */
function submit_operation_read(e) {
    // not to post
    e.preventDefault();
    var formData = $(operation_read_form).serialize();

    // read operation
    operation_read(formData);

    // read history
    history_read(formData);

    // read report
    report_read(formData);

};



/*
 * submit function when prediction button clicked to send form to the server with ajax.
 */
function submit_prediction(e) {
    // not to post
    e.preventDefault();
    var formData = $(operation_read_form).serialize();

    // prediction
    prediction(formData);

};



/*
 * Function to send formData to server using ajax and render its response to HTML
 */
function operation_read(formData) {
    $("#operation-body").addClass('loading-spinner');

    $.ajax({
      type: 'POST',
      beforeSend: function(request) {
        request.setRequestHeader("Authorization", Cookies.get("forecass_jwt"));
      },
      url: host_url + '/operation/read',
      data: formData
    })
    .done(function(response) {
        var operation_json = response.operation_json;
        // console.log(operation_json)
        $.each(operation_json, function(k, v) {
            try {
              k = k.replace(/\s/g,'');
              // console.log(k)
              var id = k

              // console.log('set to id:' + id + ', value:' + v)
              document.getElementById(id).value = v
              document.getElementById(id).disabled = false
            }
            catch(err) {
              console.log('error with:' + k + ', err:' + err)
            }
        });

        // check default
        if (document.getElementById('test_only_mode').value == 'true') {
           document.getElementById('test_only_mode').checked = true
        } else {
           document.getElementById('test_only_mode').checked = false
        }
        var test_only_mode = (document.getElementById('test_only_mode').checked)
        console.log('test_only_mode:' + test_only_mode)
        if (test_only_mode) {
            document.getElementById('mode_train_predict').checked = false;
        } else {
            document.getElementById('mode_train_predict').checked = true;
        }
        document.getElementById('mode_train_predict').disabled = false;

        // has_batch_norm
        document.getElementById('has_batch_norm').checked = (document.getElementById('has_batch_norm').value == 'true')
        // has_res_net
        document.getElementById('has_res_net').checked = (document.getElementById('has_res_net').value == 'true')
        // has_to_complement_before
        document.getElementById('has_to_complement_before').checked = (document.getElementById('has_to_complement_before').value == 'true')
        // add_l1_norm_reg
        document.getElementById('add_l1_norm_reg').checked = (document.getElementById('add_l1_norm_reg').value == 'true')
        // has_res_net
        document.getElementById('has_res_net').checked = (document.getElementById('has_res_net').value == 'true')
        // skip_invalid_data
        document.getElementById('skip_invalid_data').checked = (document.getElementById('skip_invalid_data').value == 'true')
        // calc_cc_errors
        document.getElementById('calc_cc_errors').checked = (document.getElementById('calc_cc_errors').value == 'true')
        // prioritize_cloud
        document.getElementById('prioritize_cloud').checked = (document.getElementById('prioritize_cloud').value == 'true')

        toggle_elements_by_read_status(have_read=true)
        console.log('ajax success with message:' + response.message)


    })
    .fail(function(response) {
        console.log('ajax fail with message:' + response.message)
    })
    .always(function(response) {
        console.log("ajax complete");
        // $("#setting-loading").removeClass('loading-spinner');
        $("#operation-body").removeClass('loading-spinner');
    });

};

/*
 * Function to send formData to server using ajax and render its response to HTML
 */
function report_read(formData) {
    $("#operation-body").addClass('loading-spinner');

    // Report read
    $.ajax({
      type: 'POST',
      url: host_url + '/report/read',
      data: formData
    })
    .done(function(response) {
        var file_tree = response.file_tree;
        // console.log(file_tree)

        function loop_children(item, parent_id, this_id) {
            $('#report_file_tree_' + parent_id).append('<li id=\"report_file_tree_' + this_id + '\">' + item.name + '</li>')
            console.log('parent_id:' + parent_id + ' this_id:' + this_id + ' item:' + item)
            if (item.children) {
                var child_id = this_id
                $('#report_file_tree_' + this_id).append('<ul id=\"report_file_tree_' + child_id + '\"></ul>')
                console.log('parent_id:' + parent_id + ' this_id:' + this_id + ' has children, child_id:' + child_id)
                 for (key in item.children) {
                     console.log('child_id:' + child_id + ', key:' + key + ', value:' + item.children[key])
                     child_id = child_id + 1
                     loop_children(item.children[key], this_id, child_id)
                }
            } else {
                console.log('parent_id:' + parent_id + ' this_id:' + this_id + ' no children, name:' + item.name)
                $('#report_file_tree_' + this_id).append('<a href=\"' + item.link + '\">[download]</li>')
            }
        };

        $('#report_file_tree_0').empty();
        loop_children(file_tree, 0, 1);

        console.log('report/read message:' + response.message)
        console.log('report/read responseText:' + response.responseText)


    })
    .always(function(response) {
        console.log("ajax complete");
        // $("#report-loading").removeClass('loading-spinner');
        $("#operation-body").removeClass('loading-spinner');

    })
    .fail(function(response) {
        console.log('report/read message:' + response.message)
        console.log('report/read responseText:' + response.responseText)
        console.log('report/read responseText.message:' + response.responseText.message)
    });

};

/*
 * Function to send formData to server using ajax and render its response to HTML
 */
function history_read(formData) {
    $("#operation-body").addClass('loading-spinner');

    // Report read
    $.ajax({
      type: 'POST',
      url: host_url + '/history/read',
      data: formData
    })
    .done(function(response) {
        var accuracy_csv = response.accuracy_csv;
        // console.log('accuracy_csv:' + accuracy_csv)

        console.log('done history/read message:' + response.message)
        console.log('done history/read responseText:' + response.responseText)

        var char_tag_id = 'history_accuracy'
        if (!(document.getElementById(char_tag_id) instanceof Object)) {
            $('#tabs_history').append('<div id=\"' + char_tag_id + '\"></div>')
        }

        var plotly_data = convert_array_to_plotly_data(accuracy_csv);
        draw_with_plotly('accuracy', plotly_data, char_tag_id);


    })
    .always(function(response) {
        console.log("ajax complete");
        // $("#history-loading").removeClass('loading-spinner');
        $("#operation-body").removeClass('loading-spinner');


    })
    .fail(function(response) {
        console.log('fail history/read message:' + response.message)
        console.log('fail history/read responseText:' + response.responseText)
        console.log('fail history/read responseText.message:' + response.responseText.message)
    });

};

function convert_array_to_chart_js_data(data) {

  var _labels = [], _train_history_data = [], _test_history_data = [];
  for (var row in data) {
    _labels.push(data[row][0])
    _test_history_data.push(data[row][2])
    _train_history_data.push(data[row][3])
  };

  chart_js_data = {
    type: 'line',
    data: {
      labels: _labels,
      datasets: [
        { label: "test", data: _test_history_data, borderColor: "red"},
        { label: "train", data: _train_history_data, borderColor: "blue",}
      ]
    },
    options: {
      title: {
        display: true,
        text: 'accuracy'
      }
    }
  };
  return chart_js_data
};

function convert_array_to_plotly_data(data) {
    var _labels = [], _train_history_data = [], _test_history_data = [];
    for (var row in data) {
        _labels.push(data[row][0])
        _test_history_data.push(data[row][2])
        _train_history_data.push(data[row][3])
    };

    var test_history_data = {
      type: 'line',
      name: "test",
      x: _labels,
      y: _test_history_data,
      mode: 'lines+markers'
    };

    var train_history_data = {
      type: 'line',
      name: "train",
      x: _labels,
      y: _train_history_data,
      mode: 'lines+markers'
    };

    console.log('test_history_data:' + test_history_data)

    return [test_history_data, train_history_data];
}

function draw_with_chart_js(chart_js_data, char_tag_id) {
  var ctx = document.getElementById(char_tag_id).getContext("2d");
  var myChart = new Chart(ctx, chart_js_data);
}

function draw_with_plotly(title, plotly_data, char_tag_id) {
    console.log('todo draw_with_plotly')

    var layout = {
        title:title
    };

    var otherSettings = {
        responsive: true,
        displayModeBar: false
    }

    Plotly.newPlot(char_tag_id, plotly_data, layout, otherSettings);
    console.log('done draw_with_plotly')

}


/*
 * Function to send formData to server using ajax and render its response to HTML
 */
function prediction(formData) {
    $("#operation-body").addClass('loading-spinner');

    // Report read
    $.ajax({
      type: 'POST',
      url: host_url + '/prediction',
      data: formData
    })
    .done(function(response) {
        var prediction_operation_file_path = response.prediction_operation_file_path;
        console.log('prediction_operation_file_path:' + prediction_operation_file_path)

        console.log('done prediction/read message:' + response.message)
        console.log('done prediction/read responseText:' + response.responseText)



    })
    .always(function(response) {
        console.log("ajax complete");
        // $("#prediction-loading").removeClass('loading-spinner');
        $("#operation-body").removeClass('loading-spinner');

    })
    .fail(function(response) {
        console.log('fail prediction/read message:' + response.message)
        console.log('fail prediction/read responseText:' + response.responseText)
        console.log('fail prediction/read responseText.message:' + response.responseText.message)
    });

};

document.getElementById('operation_read_submit').addEventListener('click', submit_operation_read, false);
document.getElementById('prediction_submit').addEventListener('click', submit_prediction, false);

document.getElementById('report_reload_submit').addEventListener('click', function(e) {
    // not to post
    e.preventDefault();
    var formData = $(operation_read_form).serialize();
    // read report
    report_read(formData);
});

document.getElementById('history_reload_submit').addEventListener('click', function(e) {
    // not to post
    e.preventDefault();
    var formData = $(operation_read_form).serialize();
    // read history
    history_read(formData);
});



/*
 * submit function when cancel button clicked.
 */
function submit_cancel(e) {
    // not to post
    e.preventDefault();
    // cancel proc
    toggle_elements_by_read_status(has_read=false)
;

};

document.getElementById('cancel_submit').addEventListener('click', submit_cancel, false);

// history_read_form.addEventListener('change', submit_history_read, false);

// for i18next change language

$('.lang_to_set').click(function () {
var lang_to_set = $(this).attr('data-lang');
console.log(lang_to_set)
i18next
.use(i18nextXHRBackend) // use locale setting json
.init(
{
    lng: lang_to_set // update setting lang
}, function(err, t) {
    jqueryI18next.init(i18next, $);
    $("[data-i18n]").localize();
});

});

// for BULMA css navbar-burger
document.addEventListener('DOMContentLoaded', () => {

  // Get all "navbar-burger" elements
  const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  // Check if there are any navbar burgers
  if ($navbarBurgers.length > 0) {

    // Add a click event on each of them
    $navbarBurgers.forEach( el => {
      el.addEventListener('click', () => {

        // Get the target from the "data-target" attribute
        const target = el.dataset.target;
        const $target = document.getElementById(target);

        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');

      });
    });
  }

});