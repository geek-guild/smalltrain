
// setting host of API server
var host_url = '.'; // for local

// Form oject
var signin_form = document.getElementById('signin_form');
// Add submit event to signin button
document.getElementById('signin_form_submit').addEventListener('click', submit_signin, false);
$('#signin_message').removeClass('is-danger')
$('#signin_message').text('')

/*
 * submit function from the form and sends it to the server with ajax.
 */
function submit_signin(e) {
    console.log('submit_signin')
    $('#signin_message').removeClass('is-danger')
    $('#signin_message').text('')

    // not to post
    e.preventDefault();
    // var form_object = $(signin_form);
    var form_serialized = $(signin_form).serializeArray();
    console.log('form_serialized: ' + form_serialized)
    // read operation
    signin(form_to_json(form_serialized));
}

function form_to_json(form_serialized) {
    console.log('form_to_json')
    var json_object = {};
    form_serialized.forEach((value, key) => {json_object[value.name] = value.value});
  return json_object;
}

/*
 * Function to send formData to server using ajax and render its response to HTML
 */
function signin(json_object) {
    //$("#operation-body").addClass('loading-spinner');
    console.log('signin')
    console.log('json_object: ' + JSON.stringify(json_object))

    $.ajax({
      type: 'POST',
      url: host_url + '/signin',
      data: JSON.stringify(json_object),
      contentType: 'application/json',
      dataType: "json"
    })
    .done(function(response) {
        var jwt = response.data.jwt;
        console.log(jwt)
        Cookies.set("forecass_jwt", jwt)
        var redirect_to = "./forecass_web.html"
        location.href = redirect_to;


    })
    .fail(function(response) {
        console.log('ajax fail with message:' + response.message)
        $('#signin_message').addClass('is-danger')
        $('#signin_message').text(response.message)
    })
    .always(function(response) {
        console.log("ajax complete");
//        $("#operation-body").removeClass('loading-spinner');
    });

};

