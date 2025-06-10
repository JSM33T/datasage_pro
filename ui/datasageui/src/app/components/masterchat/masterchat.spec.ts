import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Masterchat } from './masterchat';

describe('Masterchat', () => {
  let component: Masterchat;
  let fixture: ComponentFixture<Masterchat>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Masterchat]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Masterchat);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
