// 主要JavaScript功能
$(document).ready(function() {
    console.log('电影推荐系统初始化完成');
    
    // 从localStorage加载用户ID
    var savedUserId = localStorage.getItem('userId');
    if (savedUserId) {
        $('#user-id').val(savedUserId);
    }

    // 获取推荐
    $('#get-recommendations').click(function() {
        var userId = $('#user-id').val();
        
        if (!userId) {
            alert('请输入用户ID');
            return;
        }
        
        $('#recommendations-loading').show();
        $('#recommendations-results').hide();
        $('#initial-state').hide();
        
        $.ajax({
            url: '/api/recommendations/get_recommendations/',
            method: 'GET',
            data: {
                'user_id': userId
            },
            success: function(response) {
                displayRecommendations(response);
            },
            error: function(xhr) {
                alert('获取推荐失败: ' + (xhr.responseJSON?.error || '网络错误'));
            },
            complete: function() {
                $('#recommendations-loading').hide();
            }
        });
    });
    
    // 刷新推荐
    $('#refresh-recommendations').click(function() {
        var userId = $('#user-id').val();
        
        if (!userId) {
            alert('请输入用户ID');
            return;
        }
        
        $.ajax({
            url: '/api/recommendations/refresh_recommendations/',
            method: 'POST',
            data: {
                'user_id': userId
            },
            success: function(response) {
                displayRecommendations(response);
                showAlert('推荐已刷新！', 'success');
            },
            error: function(xhr) {
                alert('刷新推荐失败: ' + (xhr.responseJSON?.error || '网络错误'));
            }
        });
    });
});

// 修复：评分功能使用新的事件绑定和UI更新
function rateMovie(userId, movieId, rating, $element) {
    if (!userId) {
        alert('请先设置用户ID');
        return;
    }
    
    console.log('评分:', rating, '电影:', movieId);
    
    // 立即更新UI反馈
    updateRatingUI(movieId, rating, $element);
    
    // 显示加载状态
    $element.prop('disabled', true).html('<i class="fa fa-spinner fa-spin"></i>');
    
    $.ajax({
        url: '/api/ratings/rate_movie/',
        method: 'POST',
        data: {
            'user_id': userId,
            'movie_id': movieId,
            'rating': rating
        },
        success: function(response) {
            console.log('评分成功:', response);
            showAlert(`成功评分 ${rating} 星！`, 'success');
            
            // 更新评分显示区域
            $(`#rating-${movieId}`).html(
                `<span class="text-success" style="font-weight: 600;"><i class="fa fa-check-circle"></i> 已评分: ${rating}星</span>`
            );
        },
        error: function(xhr) {
            console.error('评分失败:', xhr.responseJSON);
            showAlert('评分失败: ' + (xhr.responseJSON?.error || '网络错误'), 'error');
            // 出错时恢复按钮状态
            resetRatingButtons($element.closest('.rating-controls'));
        },
        complete: function() {
            $element.prop('disabled', false).html(rating + ' <i class="fa fa-star"></i>');
        }
    });
}

// 修复：更新评分按钮UI状态
function updateRatingUI(movieId, selectedRating, $clickedButton) {
    const $ratingControls = $clickedButton.closest('.rating-controls');
    const $allButtons = $ratingControls.find('.btn-rating');
    
    console.log('更新UI: 选中评分', selectedRating, '按钮数量:', $allButtons.length);
    
    // 移除所有激活状态
    $allButtons.removeClass('active rated');
    
    // 为选中的评分及之前的所有按钮添加激活状态
    $allButtons.each(function() {
        const $btn = $(this);
        const btnRating = parseInt($btn.data('rating'));
        
        console.log('检查按钮:', btnRating, '是否<=', selectedRating);
        
        if (btnRating <= selectedRating) {
            $btn.addClass('active');
            if (btnRating === selectedRating) {
                $btn.addClass('rated');
            }
        }
    });
    
    // 调试：检查最终状态
    setTimeout(() => {
        const activeButtons = $ratingControls.find('.btn-rating.active');
        console.log('激活按钮数量:', activeButtons.length, '应该为:', selectedRating);
    }, 100);
}

// 重置评分按钮状态
function resetRatingButtons($ratingControls) {
    const $allButtons = $ratingControls.find('.btn-rating');
    $allButtons.removeClass('active rated');
}

// 显示提示信息
function showAlert(message, type) {
    // 移除现有的提示
    $('.custom-alert').remove();
    
    var alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    var icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-triangle';
    
    var $alert = $(
        `<div class="alert ${alertClass} alert-dismissible custom-alert" style="position: fixed; top: 80px; right: 20px; z-index: 9999; min-width: 300px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <button type="button" class="close" data-dismiss="alert" style="position: absolute; right: 10px; top: 10px;">&times;</button>
            <div style="padding-right: 25px;">
                <i class="fa ${icon}"></i> <strong>${type === 'success' ? '成功!' : '错误!'}</strong> ${message}
            </div>
        </div>`
    );
    
    $('body').append($alert);
    
    // 自动消失
    setTimeout(function() {
        $alert.fadeOut(300, function() {
            $(this).alert('close');
        });
    }, 3000);
}

// 修复：显示推荐结果的函数
function displayRecommendations(data) {
    var $results = $('#recommendations-results');
    var $container = $results.find('.recommendations-container');
    
    $container.empty();
    
    if (data.recommended_movies_details && data.recommended_movies_details.length > 0) {
        data.recommended_movies_details.forEach(function(movie, index) {
            // 确定电影类型类名
            var typeClass = 'default';
            if (movie.genres) {
                if (movie.genres.includes('Comedy')) typeClass = 'comedy';
                else if (movie.genres.includes('Action')) typeClass = 'action';
                else if (movie.genres.includes('Drama')) typeClass = 'drama';
                else if (movie.genres.includes('Romance')) typeClass = 'romance';
                else if (movie.genres.includes('Horror')) typeClass = 'horror';
                else if (movie.genres.includes('Children') || movie.genres.includes('Animation')) typeClass = 'children';
                else if (movie.genres.includes('Sci-Fi')) typeClass = 'scifi';
                else if (movie.genres.includes('Documentary')) typeClass = 'documentary';
            }
            
            var movieHtml = `
                <div class="col-sm-6 col-md-4 col-lg-3">
                    <div class="movie-card">
                        <div class="movie-poster ${typeClass}">
                            <i class="fa fa-film"></i>
                        </div>
                        <div class="movie-info">
                            <div class="movie-title">${movie.title}</div>
                            <div class="movie-genres">
                                ${movie.genres ? movie.genres.split('|').slice(0, 3).map(genre => 
                                    `<span class="genre-tag">${genre}</span>`
                                ).join('') : '<span class="genre-tag">未知类型</span>'}
                            </div>
                            <div class="recommendation-badge">推荐 #${index + 1}</div>
                            <div class="rating-section">
                                <div id="rating-${movie.movie_id}" class="text-muted small" style="margin-bottom: 8px;"></div>
                                <div class="rating-controls">
                                    <small class="rating-label">评分:</small>
                                    ${[1,2,3,4,5].map(star => `
                                        <button class="btn btn-rating" 
                                                data-rating="${star}" data-movie-id="${movie.movie_id}">
                                            ${star} <i class="fa fa-star"></i>
                                        </button>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            $container.append(movieHtml);
        });
    } else {
        $container.html('<div class="col-12"><div class="alert alert-info">暂无推荐结果</div></div>');
    }
    
    $results.show();
    
    // 重新绑定评分按钮事件 - 使用新的绑定方式
    $('.btn-rating').off('click').on('click', function() {
        var rating = $(this).data('rating');
        var movieId = $(this).data('movie-id');
        var userId = $('#user-id').val() || localStorage.getItem('userId') || 1;
        
        console.log('动态绑定点击: 电影', movieId, '评分', rating);
        rateMovie(userId, movieId, rating, $(this));
    });
}

// 全局绑定评分按钮事件（用于静态加载的页面）
$(document).on('click', '.btn-rating', function() {
    var rating = $(this).data('rating');
    var movieId = $(this).data('movie-id');
    var userId = $('#user-id').val() || localStorage.getItem('userId') || 1;
    
    console.log('全局点击: 电影', movieId, '评分', rating);
    rateMovie(userId, movieId, rating, $(this));
});

// 首页设置用户ID功能
$(document).on('click', '#set-user-id', function() {
    var userId = $('#user-id-input').val();
    if (userId) {
        localStorage.setItem('userId', userId);
        showAlert('用户ID已设置为: ' + userId, 'success');
        setTimeout(function() {
            window.location.href = "/recommendations/";
        }, 1000);
    } else {
        showAlert('请输入有效的用户ID', 'error');
    }
});

// 个人中心加载评分记录
$(document).on('click', '#load-profile', function() {
    var userId = $('#user-id').val();
    if (!userId) {
        alert('请输入用户ID');
        return;
    }

    localStorage.setItem('userId', userId);
    loadUserRatings(userId);
});

function loadUserRatings(userId) {
    $('#ratings-loading').show();
    
    $.ajax({
        url: '/api/user/' + userId + '/ratings/',
        method: 'GET',
        success: function(ratings) {
            displayUserRatings(ratings);
            updateStats(ratings);
        },
        error: function(xhr) {
            alert('加载评分记录失败');
        },
        complete: function() {
            $('#ratings-loading').hide();
        }
    });
}

function displayUserRatings(ratings) {
    var $container = $('#ratings-container');
    
    if (ratings.length === 0) {
        $container.html('<div class="alert alert-info">暂无评分记录</div>');
        return;
    }

    var html = '<div class="list-group">';
    ratings.forEach(function(rating) {
        var stars = '';
        for (var i = 1; i <= 5; i++) {
            var active = i <= rating.rating ? 'active' : '';
            stars += `<span class="star ${active}">★</span>`;
        }
        
        html += `
            <div class="list-group-item">
                <div class="row">
                    <div class="col-md-8">
                        <h5 class="list-group-item-heading">${rating.movie_title}</h5>
                        <p class="list-group-item-text text-muted">${rating.movie_genres || '未知类型'}</p>
                    </div>
                    <div class="col-md-4 text-right">
                        <div class="rating-stars" style="color: #ffc107; font-size: 18px;">
                            ${stars}
                        </div>
                        <small class="text-muted">${new Date(rating.timestamp).toLocaleDateString()}</small>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    $container.html(html);
}

function updateStats(ratings) {
    $('#total-ratings').text(ratings.length);
    $('#rated-movies').text(ratings.length);
    
    if (ratings.length > 0) {
        var total = ratings.reduce(function(sum, rating) {
            return sum + rating.rating;
        }, 0);
        var avg = (total / ratings.length).toFixed(1);
        $('#avg-rating').text(avg);
    }
}